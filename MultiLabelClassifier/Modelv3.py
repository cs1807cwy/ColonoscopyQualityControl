import json
import os
import shutil
import time

import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple

import torchvision.utils
from lightning.pytorch import LightningModule

from .Network import *


class MultiLabelClassifier_ViT_L_Patch14_336_Class7(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (336, 336),  # must be 336*336 as we use pretrained ViT_Patch14_336
            num_heads: int = 8,  # heads number in [1, 2, 4, 6, 8]
            attention_lambda: float = 0.3,
            num_classes: int = 7,
            thresh: float = 0.5,
            batch_size: int = 16,
            lr: float = 0.0001,
            epochs: int = 1000,
            momentum: float = 0.9,
            weight_decay: float = 0.0001,
            cls_weight: float = 4.,
            outside_acc_thresh: float = 0.9,
            nonsense_acc_thresh: float = 0.9,
            data_root: str = None,
            test_id_map_file_path: str = None,
            test_viz_save_dir: str = 'test_viz',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.thresh = thresh
        self.input_shape = input_shape

        # networks
        self.backbone = ViT_L_Patch14_336_Extractor(True)
        self.classify_head = ClassSpecificMultiHeadAttention(num_heads, attention_lambda, 1024, num_classes)  # embed dim=1024
        self.example_input_array = torch.zeros(batch_size, 3, input_shape[0], input_shape[1])

        # for viz
        self.index_label: Dict[int, str] = {
            0: 'outside',
            1: 'nonsense',
            2: 'ileocecal',
            3: 'bbps0',
            4: 'bbps1',
            5: 'bbps2',
            6: 'bbps3',
        }

        self.outside_acc_thresh = outside_acc_thresh
        self.nonsense_acc_thresh = nonsense_acc_thresh
        self.confuse_matrix: Dict = {}

        self.idmap = None
        self.fps_timer = None
        self.count = None

    def forward(self, feature):
        feature = self.backbone(feature)

        # (B, 1+HW, C)
        # we use all the feature to form the tensor like B C H W
        feature = feature[:, 1:]
        b, hw, c = feature.shape
        h = w = int(math.sqrt(hw))
        feature = feature.transpose(1, 2).contiguous()
        feature = feature.reshape(b, c, h, w)
        logit = self.classify_head(feature)

        return logit

    def training_step(self, batch, batch_idx: int):
        image, label_gt = batch
        pred = self(image)
        loss, loss_loc, loss_cls = self._calculate_loss(pred, label_gt)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_loc', loss_loc, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_cls', loss_cls, prog_bar=True, logger=True, sync_dist=True)

        # 计算总体train_mean_acc
        label_pred_tf = torch.ge(F.sigmoid(pred), self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = float(torch.eq(label_pred_tf, label_gt_tf).float().mean().cpu())
        self.log(f'train_thresh_mean_acc', mean_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def _calculate_loss(self, pred, gt):
        # 逐标签二元交叉熵损失
        loss_loc = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean')
        # 清洁度交叉熵损失
        loss_cls = F.cross_entropy(pred[:, 3:], gt[:, 3:], reduction='mean')
        return loss_loc + self.hparams.cls_weight * loss_cls, loss_loc, loss_cls

    def configure_optimizers(self):
        return self._configure_sgd_cosine()

    def _configure_sgd_cosine(self):
        # optimizer and warmup
        backbone, classifier = [], []
        for name, param in self.named_parameters():
            if 'classify_head' in name:
                classifier.append(param)
            else:
                backbone.append(param)
        optimizer = torch.optim.SGD(
            [
                {'params': backbone, 'lr': self.hparams.lr},
                {'params': classifier, 'lr': self.hparams.lr * 10.}
            ],
            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return [optimizer], [scheduler]

    def on_validation_epoch_start(self):
        self.confuse_matrix: Dict[str, float] = {}
        for i in range(3):
            self.confuse_matrix[f'label_{self.index_label[i]}_TP'] = 0.
            self.confuse_matrix[f'label_{self.index_label[i]}_FP'] = 0.
            self.confuse_matrix[f'label_{self.index_label[i]}_FN'] = 0.
            self.confuse_matrix[f'label_{self.index_label[i]}_TN'] = 0.
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 3]}_gt_{self.index_label[j + 3]}'] = 0.
        self.confuse_matrix['label_cleansing_low_TP'] = 0.
        self.confuse_matrix['label_cleansing_low_FP'] = 0.
        self.confuse_matrix['label_cleansing_low_FN'] = 0.
        self.confuse_matrix['label_cleansing_low_TN'] = 0.

    def validation_step(self, batch, batch_idx: int):
        image, label_gt = batch
        logit = F.sigmoid(self(image))

        # 计算val_acc
        # label_pred_tf: BoolTensor[B, 7] = B * [nonsense?, outside?, ileocecal?, bbps0?, bbps1?, bbps2?, bbps3?]
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = float(torch.eq(label_pred_tf, label_gt_tf).float().mean().cpu())
        self.log('val_thresh_mean_acc', mean_acc, prog_bar=True, logger=True, sync_dist=True)

        # 体内外logit: FloatTensor[B]
        in_out_logit = logit[:, 0]
        # 体内外标签: BoolTensor[B]
        # outside时为True
        label_in_out_pred = torch.ge(in_out_logit, self.hparams.thresh)
        # 体内外gt: BoolTensor[B]
        label_in_out_gt = torch.ge(label_gt[:, 0], self.hparams.thresh)
        self.confuse_matrix[f'label_{self.index_label[0]}_TP'] += float((label_in_out_pred & label_in_out_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[0]}_FP'] += float((label_in_out_pred & ~label_in_out_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[0]}_FN'] += float((~label_in_out_pred & label_in_out_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[0]}_TN'] += float((~label_in_out_pred & ~label_in_out_gt).float().sum().cpu())

        # 帧质量logit: FloatTensor[B]
        nonsense_logit = logit[:, 1]
        # 坏帧标签: BoolTensor[B]
        # nonsense时为True
        label_nonsense_pred = torch.ge(nonsense_logit, self.hparams.thresh)
        # 帧质量gt: BoolTensor[B]
        # pred或gt是outside时不计入总数
        label_nonsense_gt = torch.ge(label_gt[:, 1], self.hparams.thresh)
        flag = ~label_in_out_pred & ~label_in_out_gt
        self.confuse_matrix[f'label_{self.index_label[1]}_TP'] += float((flag & label_nonsense_pred & label_nonsense_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[1]}_FP'] += float((flag & label_nonsense_pred & ~label_nonsense_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[1]}_FN'] += float((flag & ~label_nonsense_pred & label_nonsense_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[1]}_TN'] += float((flag & ~label_nonsense_pred & ~label_nonsense_gt).float().sum().cpu())

        # 回盲部logit: FloatTensor[B]
        ileo_logit = logit[:, 2]
        # 回盲部标签: BoolTensor[B]
        label_ileo_pred = torch.ge(ileo_logit, self.hparams.thresh)
        # 回盲部gt: BoolTensor[B]
        label_ileo_gt = torch.ge(label_gt[:, 2], self.hparams.thresh)
        flag = ~label_in_out_pred & ~label_in_out_gt & ~label_nonsense_pred & ~label_nonsense_gt
        self.confuse_matrix[f'label_{self.index_label[2]}_TP'] += float((flag & label_ileo_pred & label_ileo_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[2]}_FP'] += float((flag & label_ileo_pred & ~label_ileo_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[2]}_FN'] += float((flag & ~label_ileo_pred & label_ileo_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[2]}_TN'] += float((flag & ~label_ileo_pred & ~label_ileo_gt).float().sum().cpu())

        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 3:]
        # 清洁度label: IntTensor[B] (取预测值最大的，但会被outside标签抑制)
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度gt: IntTensor[B]
        label_cls_gt = torch.argmax(label_gt[:, 3:], dim=-1)
        flag = ~label_in_out_pred & ~label_in_out_gt & ~label_nonsense_pred & ~label_nonsense_gt
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 3]}_gt_{self.index_label[j + 3]}'] += \
                    float((flag & torch.eq(label_cls_pred, i) & torch.eq(label_cls_gt, j)).float().sum().cpu())  # flag用于清洁度标签抑制

    def on_validation_epoch_end(self):
        metrics: Dict[str, float] = {}

        # 体内外
        TP: float = self.confuse_matrix[f'label_{self.index_label[0]}_TP']
        FP: float = self.confuse_matrix[f'label_{self.index_label[0]}_FP']
        FN: float = self.confuse_matrix[f'label_{self.index_label[0]}_FN']
        TN: float = self.confuse_matrix[f'label_{self.index_label[0]}_TN']
        metrics[f'label_{self.index_label[0]}_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.

        # 帧质量
        TP: float = self.confuse_matrix[f'label_{self.index_label[1]}_TP']
        FP: float = self.confuse_matrix[f'label_{self.index_label[1]}_FP']
        FN: float = self.confuse_matrix[f'label_{self.index_label[1]}_FN']
        TN: float = self.confuse_matrix[f'label_{self.index_label[1]}_TN']
        metrics[f'label_{self.index_label[1]}_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.

        # 回盲部
        TP: float = self.confuse_matrix[f'label_{self.index_label[2]}_TP']
        FP: float = self.confuse_matrix[f'label_{self.index_label[2]}_FP']
        FN: float = self.confuse_matrix[f'label_{self.index_label[2]}_FN']
        TN: float = self.confuse_matrix[f'label_{self.index_label[2]}_TN']
        prec_nothresh = TP / (TP + FP) if TP + FP > 0. else 0.
        prec_thresh = prec_nothresh \
            if metrics[f'label_{self.index_label[0]}_acc'] > self.outside_acc_thresh \
               and metrics[f'label_{self.index_label[1]}_acc'] > self.nonsense_acc_thresh \
            else 0.
        metrics[f'label_{self.index_label[2]}_prec_nothresh'] = prec_nothresh
        metrics[f'label_{self.index_label[2]}_prec_thresh'] = prec_thresh

        acc_nothresh = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.
        acc_thresh = acc_nothresh \
            if metrics[f'label_{self.index_label[0]}_acc'] > self.outside_acc_thresh \
               and metrics[f'label_{self.index_label[1]}_acc'] > self.nonsense_acc_thresh \
            else 0.
        metrics[f'label_{self.index_label[2]}_acc_nothresh'] = acc_nothresh
        metrics[f'label_{self.index_label[2]}_acc_thresh'] = acc_thresh

        # 四分清洁度准确率
        total: float = 0.
        correct: float = 0.
        for i in range(3, 7):  # i: predict
            for j in range(3, 7):  # j: gt
                tmp = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i]}_gt_{self.index_label[j]}']
                if i == j:
                    correct += tmp
                total += tmp
        acc_nothresh = correct / total if total > 0. else 0.
        acc_thresh = acc_nothresh \
            if metrics[f'label_{self.index_label[0]}_acc'] > self.outside_acc_thresh \
               and metrics[f'label_{self.index_label[1]}_acc'] > self.nonsense_acc_thresh \
            else 0.
        metrics[f'label_cleansing_acc_nothresh'] = acc_nothresh
        metrics[f'label_cleansing_acc_thresh'] = acc_thresh

        # bbps0-1/bbps2-3二分清洁度准确率
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                cnt = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 3]}_gt_{self.index_label[j + 3]}']
                if i in {0, 1} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_TP'] += cnt
                elif i in {0, 1} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_FP'] += cnt
                elif i in {2, 3} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_FN'] += cnt
                elif i in {2, 3} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_TN'] += cnt
        TP: float = self.confuse_matrix['label_cleansing_low_TP']
        FP: float = self.confuse_matrix['label_cleansing_low_FP']
        FN: float = self.confuse_matrix['label_cleansing_low_FN']
        TN: float = self.confuse_matrix['label_cleansing_low_TN']
        acc_nothresh = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.
        acc_thresh = acc_nothresh \
            if metrics[f'label_{self.index_label[0]}_acc'] > self.outside_acc_thresh \
               and metrics[f'label_{self.index_label[1]}_acc'] > self.nonsense_acc_thresh \
            else 0.
        metrics['label_cleansing_biclassify_acc_nothresh'] = acc_nothresh
        metrics['label_cleansing_biclassify_acc_thresh'] = acc_thresh

        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)
        self.log_dict(metrics, logger=True, sync_dist=True)

    def on_test_epoch_start(self):
        # 启动计时
        self.fps_timer = time.perf_counter()
        self.count = 0

        if self.hparams.test_id_map_file_path is not None:
            with open(self.hparams.test_id_map_file_path, 'r') as fp:
                self.idmap: Dict[str, str] = json.load(fp)['idmap']

        self.confuse_matrix: Dict[str, float] = {}
        for i in range(3):
            self.confuse_matrix[f'label_{self.index_label[i]}_TP'] = 0.
            self.confuse_matrix[f'label_{self.index_label[i]}_FP'] = 0.
            self.confuse_matrix[f'label_{self.index_label[i]}_FN'] = 0.
            self.confuse_matrix[f'label_{self.index_label[i]}_TN'] = 0.
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 3]}_gt_{self.index_label[j + 3]}'] = 0.
        self.confuse_matrix['label_cleansing_low_TP'] = 0.
        self.confuse_matrix['label_cleansing_low_FP'] = 0.
        self.confuse_matrix['label_cleansing_low_FN'] = 0.
        self.confuse_matrix['label_cleansing_low_TN'] = 0.

        if self.hparams.test_viz_save_dir is not None:
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_outside_gt_inside'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_inside_gt_outside'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_nonsense_gt_fine'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_fine_gt_nonsense'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_ileocecal_gt_nofeature'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_nofeature_gt_ileocecal'), exist_ok=True)
            for i in range(0, 4):  # i: predict
                for j in range(0, 4):  # j: gt
                    if i != j:
                        os.makedirs(os.path.join(self.hparams.test_viz_save_dir, f'pred_bbps{i}_gt_bbps{j}'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_lq_gt_hq'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.test_viz_save_dir, 'pred_hq_gt_lq'), exist_ok=True)

    def test_step(self, batch, batch_idx: int):
        image_id, image, label_gt = batch
        logit = F.sigmoid(self(image))

        self.count += image_id.size(0)

        # 计算test_acc
        # label_pred_tf: BoolTensor[B, 7] = B * [nonsense?, outside?, ileocecal?, bbps0?, bbps1?, bbps2?, bbps3?]
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = float(torch.eq(label_pred_tf, label_gt_tf).float().mean().cpu())
        self.log('test_thresh_mean_acc', mean_acc, prog_bar=True, logger=True, sync_dist=True)

        # 体内外logit: FloatTensor[B]
        in_out_logit = logit[:, 0]
        # 体内外标签: BoolTensor[B]
        # outside时为True
        label_in_out_pred = torch.ge(in_out_logit, self.hparams.thresh)
        # 体内外gt: BoolTensor[B]
        label_in_out_gt = torch.ge(label_gt[:, 0], self.hparams.thresh)
        self.confuse_matrix[f'label_{self.index_label[0]}_TP'] += float((label_in_out_pred & label_in_out_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[0]}_FP'] += float((label_in_out_pred & ~label_in_out_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[0]}_FN'] += float((~label_in_out_pred & label_in_out_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[0]}_TN'] += float((~label_in_out_pred & ~label_in_out_gt).float().sum().cpu())

        # 保存体内外分类错误的帧
        if self.hparams.test_viz_save_dir is not None:
            cnt = 0
            for iid, img, lb, gt in zip(image_id, image, label_in_out_pred, label_in_out_gt):
                if lb != gt:
                    dir_path: str = os.path.join(
                        self.hparams.test_viz_save_dir,
                        'pred_outside_gt_inside' if lb else 'pred_inside_gt_outside'
                    )
                    if self.idmap is None:
                        torchvision.utils.save_image(
                            img,
                            os.path.join(dir_path, f'batch_{batch_idx}_{cnt}.png'))
                    else:
                        path: str = os.path.join(self.hparams.data_root, self.idmap[str(int(iid.cpu()))])
                        shutil.copyfile(
                            path,
                            os.path.join(dir_path, os.path.basename(path))
                        )
                cnt += 1

        # 帧质量logit: FloatTensor[B]
        nonsense_logit = logit[:, 1]
        # 坏帧标签: BoolTensor[B]
        # nonsense时为True
        label_nonsense_pred = torch.ge(nonsense_logit, self.hparams.thresh)
        # 帧质量gt: BoolTensor[B]
        # pred或gt是outside时不计入总数
        label_nonsense_gt = torch.ge(label_gt[:, 1], self.hparams.thresh)
        flag = ~label_in_out_pred & ~label_in_out_gt
        self.confuse_matrix[f'label_{self.index_label[1]}_TP'] += float((flag & label_nonsense_pred & label_nonsense_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[1]}_FP'] += float((flag & label_nonsense_pred & ~label_nonsense_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[1]}_FN'] += float((flag & ~label_nonsense_pred & label_nonsense_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[1]}_TN'] += float((flag & ~label_nonsense_pred & ~label_nonsense_gt).float().sum().cpu())

        # 保存帧质量分类错误的帧
        if self.hparams.test_viz_save_dir is not None:
            cnt = 0
            for iid, img, fg, lb, gt in zip(image_id, image, flag, label_nonsense_pred, label_nonsense_gt):
                if fg and (lb != gt):
                    dir_path: str = os.path.join(
                        self.hparams.test_viz_save_dir,
                        'pred_nonsense_gt_fine' if lb else 'pred_fine_gt_nonsense'
                    )
                    if self.idmap is None:
                        torchvision.utils.save_image(
                            img,
                            os.path.join(dir_path, f'batch_{batch_idx}_{cnt}.png'))
                    else:
                        path: str = os.path.join(self.hparams.data_root, self.idmap[str(int(iid.cpu()))])
                        shutil.copyfile(
                            path,
                            os.path.join(dir_path, os.path.basename(path))
                        )
                cnt += 1

        # 回盲部logit: FloatTensor[B]
        ileo_logit = logit[:, 2]
        # 回盲部标签: BoolTensor[B]
        label_ileo_pred = torch.ge(ileo_logit, self.hparams.thresh)
        # 回盲部gt: BoolTensor[B]
        label_ileo_gt = torch.ge(label_gt[:, 2], self.hparams.thresh)
        flag = ~label_in_out_pred & ~label_in_out_gt & ~label_nonsense_pred & ~label_nonsense_gt
        self.confuse_matrix[f'label_{self.index_label[2]}_TP'] += float((flag & label_ileo_pred & label_ileo_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[2]}_FP'] += float((flag & label_ileo_pred & ~label_ileo_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[2]}_FN'] += float((flag & ~label_ileo_pred & label_ileo_gt).float().sum().cpu())
        self.confuse_matrix[f'label_{self.index_label[2]}_TN'] += float((flag & ~label_ileo_pred & ~label_ileo_gt).float().sum().cpu())

        # 保存回盲部分类错误的帧
        if self.hparams.test_viz_save_dir is not None:
            cnt = 0
            for iid, img, fg, lb, gt in zip(image_id, image, flag, label_ileo_pred, label_ileo_gt):
                if fg and (lb != gt):
                    dir_path: str = os.path.join(
                        self.hparams.test_viz_save_dir,
                        'pred_ileocecal_gt_nofeature' if lb else 'pred_nofeature_gt_ileocecal'
                    )
                    if self.idmap is None:
                        torchvision.utils.save_image(
                            img,
                            os.path.join(dir_path, f'batch_{batch_idx}_{cnt}.png'))
                    else:
                        path: str = os.path.join(self.hparams.data_root, self.idmap[str(int(iid.cpu()))])
                        shutil.copyfile(
                            path,
                            os.path.join(dir_path, os.path.basename(path))
                        )
                cnt += 1

        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 3:]
        # 清洁度label: IntTensor[B] (取预测值最大的，但会被outside标签抑制)
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度gt: IntTensor[B]
        label_cls_gt = torch.argmax(label_gt[:, 3:], dim=-1)
        flag = ~label_in_out_pred & ~label_in_out_gt & ~label_nonsense_pred & ~label_nonsense_gt
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 3]}_gt_{self.index_label[j + 3]}'] += \
                    float((flag & torch.eq(label_cls_pred, i) & torch.eq(label_cls_gt, j)).float().sum().cpu())  # flag用于清洁度标签抑制

        # 保存清洁度分类错误的帧
        if self.hparams.test_viz_save_dir is not None:
            # 四分类
            cnt = 0
            for iid, img, fg, lb, gt in zip(image_id, image, flag, label_cls_pred, label_cls_gt):
                if fg and (lb != gt):
                    dir_path: str = os.path.join(
                        self.hparams.test_viz_save_dir,
                        f'pred_bbps{int(lb)}_gt_bbps{int(gt)}',
                    )
                    if self.idmap is None:
                        torchvision.utils.save_image(
                            img,
                            os.path.join(dir_path, f'batch_{batch_idx}_{cnt}.png'))
                    else:
                        path: str = os.path.join(self.hparams.data_root, self.idmap[str(int(iid.cpu()))])
                        shutil.copyfile(
                            path,
                            os.path.join(dir_path, os.path.basename(path))
                        )
                cnt += 1
            # 二分类
            cnt = 0
            for iid, img, fg, lb, gt in zip(image_id, image, flag, label_cls_pred, label_cls_gt):
                if not fg: continue
                lb = int(lb)
                gt = int(gt)
                if lb in {0, 1} and gt in {2, 3}:
                    dir_path: str = os.path.join(
                        self.hparams.test_viz_save_dir,
                        f'pred_lq_gt_hq',
                    )
                    if self.idmap is None:
                        torchvision.utils.save_image(
                            img,
                            os.path.join(dir_path, f'batch_{batch_idx}_{cnt}.png'))
                    else:
                        path: str = os.path.join(self.hparams.data_root, self.idmap[str(int(iid.cpu()))])
                        shutil.copyfile(
                            path,
                            os.path.join(dir_path, os.path.basename(path))
                        )
                elif lb in {2, 3} and gt in {0, 1}:
                    dir_path: str = os.path.join(
                        self.hparams.test_viz_save_dir,
                        f'pred_hq_gt_lq',
                    )
                    if self.idmap is None:
                        torchvision.utils.save_image(
                            img,
                            os.path.join(dir_path, f'batch_{batch_idx}_{cnt}.png'))
                    else:
                        path: str = os.path.join(self.hparams.data_root, self.idmap[str(int(iid.cpu()))])
                        shutil.copyfile(
                            path,
                            os.path.join(dir_path, os.path.basename(path))
                        )
                cnt += 1

    def on_test_epoch_end(self):
        metrics: Dict[str, float] = {}

        # 体内外
        TP: float = self.confuse_matrix[f'label_{self.index_label[0]}_TP']
        FP: float = self.confuse_matrix[f'label_{self.index_label[0]}_FP']
        FN: float = self.confuse_matrix[f'label_{self.index_label[0]}_FN']
        TN: float = self.confuse_matrix[f'label_{self.index_label[0]}_TN']
        metrics[f'label_{self.index_label[0]}_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.

        # 帧质量
        TP: float = self.confuse_matrix[f'label_{self.index_label[1]}_TP']
        FP: float = self.confuse_matrix[f'label_{self.index_label[1]}_FP']
        FN: float = self.confuse_matrix[f'label_{self.index_label[1]}_FN']
        TN: float = self.confuse_matrix[f'label_{self.index_label[1]}_TN']
        metrics[f'label_{self.index_label[1]}_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.

        # 回盲部
        TP: float = self.confuse_matrix[f'label_{self.index_label[2]}_TP']
        FP: float = self.confuse_matrix[f'label_{self.index_label[2]}_FP']
        FN: float = self.confuse_matrix[f'label_{self.index_label[2]}_FN']
        TN: float = self.confuse_matrix[f'label_{self.index_label[2]}_TN']
        metrics[f'label_{self.index_label[2]}_prec'] = TP / (TP + FP) if TP + FP > 0. else 0.
        metrics[f'label_{self.index_label[2]}_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.

        # 四分清洁度准确率
        total: float = 0.
        correct: float = 0.
        for i in range(3, 7):  # i: predict
            for j in range(3, 7):  # j: gt
                tmp = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i]}_gt_{self.index_label[j]}']
                if i == j:
                    correct += tmp
                total += tmp
        metrics[f'label_cleansing_acc'] = correct / total if total > 0. else 0.

        # bbps0-1/bbps2-3二分清洁度准确率
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                cnt = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 3]}_gt_{self.index_label[j + 3]}']
                if i in {0, 1} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_TP'] += cnt
                elif i in {0, 1} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_FP'] += cnt
                elif i in {2, 3} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_FN'] += cnt
                elif i in {2, 3} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_TN'] += cnt
        TP: float = self.confuse_matrix['label_cleansing_low_TP']
        FP: float = self.confuse_matrix['label_cleansing_low_FP']
        FN: float = self.confuse_matrix['label_cleansing_low_FN']
        TN: float = self.confuse_matrix['label_cleansing_low_TN']
        metrics['label_cleansing_biclassify_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0. else 0.

        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)
        self.log_dict(metrics, logger=True, sync_dist=True)

        # 终止计时，计算FPS
        self.log('mean_fps', float(self.count) / (time.perf_counter() - self.fps_timer), prog_bar=True, logger=True, sync_dist=True)

    def on_predict_epoch_start(self):
        # 启动计时
        self.fps_timer = time.perf_counter()
        self.count = 0

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        image = batch  # x是图像tensor，ox是原始图像tensor
        logit = F.sigmoid(self(image))

        self.count += image.size(0)

        # 体内外logit: FloatTensor[B]
        in_out_logit = logit[:, 0]
        # 体内外标签: BoolTensor[B]
        # outside时为True
        label_in_out_pred = torch.ge(in_out_logit, self.hparams.thresh)

        # 帧质量logit: FloatTensor[B]
        nonsense_logit = logit[:, 1]
        # 坏帧标签: BoolTensor[B]
        # nonsense时为True
        label_nonsense_pred = ~label_in_out_pred & torch.ge(nonsense_logit, self.hparams.thresh)

        # 回盲部logit: FloatTensor[B]
        ileo_logit = logit[:, 2]
        # 回盲部标签: BoolTensor[B]
        label_ileo_pred = ~label_in_out_pred & ~label_nonsense_pred & torch.ge(ileo_logit, self.hparams.thresh)

        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 3:]
        # 清洁度label: IntTensor[B] (取预测值最大的，但会被outside/nonsense标签抑制)
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度label_code_pred: BoolTensor[B, 4]
        label_cls_code_pred = (~label_in_out_pred & ~label_nonsense_pred).unsqueeze(1) \
                              & torch.ge(cls_logit, torch.max(cls_logit, dim=-1)[0].unsqueeze(1))

        # label_pred: FloatTensor[B, 7]
        label_pred = torch.cat(
            [
                label_in_out_pred.unsqueeze(1),
                label_nonsense_pred.unsqueeze(1),
                label_ileo_pred.unsqueeze(1),
                label_cls_code_pred
            ],
            dim=-1).float()

        # for DEBUG
        # for idx, cont in enumerate(zip(logit.cpu(), label_pred.cpu())):
        #     lg, lp = cont
        #     print(f'Dataset {dataloader_idx} | Batch {batch_idx} | Sample {idx}: Logit {lg.tolist()} | Label {lp.tolist()}')

        return logit, label_pred

    def on_predict_epoch_end(self):
        # 终止计时，计算FPS
        print(f'FPS: {float(self.count) / (time.perf_counter() - self.fps_timer)}')

    @torch.jit.export
    def forward_activate(self, batch):
        image = batch
        image = F.interpolate(image, self.input_shape, mode='bilinear', antialias=True)
        logit = F.sigmoid(self(image))

        # 体内外logit: FloatTensor[B]
        in_out_logit = logit[:, 0]
        # 体内外标签: BoolTensor[B]
        # outside时为True
        label_in_out_pred = torch.ge(in_out_logit, self.thresh)

        # 帧质量logit: FloatTensor[B]
        nonsense_logit = logit[:, 1]
        # 坏帧标签: BoolTensor[B]
        # nonsense时为True
        label_nonsense_pred = ~label_in_out_pred & torch.ge(nonsense_logit, self.thresh)

        # 回盲部logit: FloatTensor[B]
        ileo_logit = logit[:, 2]
        # 回盲部标签: BoolTensor[B]
        label_ileo_pred = ~label_in_out_pred & ~label_nonsense_pred & torch.ge(ileo_logit, self.thresh)

        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 3:]
        # 清洁度label: IntTensor[B] (取预测值最大的，但会被outside/nonsense标签抑制)
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度label_code_pred: BoolTensor[B, 4]
        label_cls_code_pred = (~label_in_out_pred & ~label_nonsense_pred).unsqueeze(1) \
                              & torch.ge(cls_logit, torch.max(cls_logit, dim=-1)[0].unsqueeze(1))

        # label_pred: FloatTensor[B, 7]
        label_pred = torch.cat(
            [
                label_in_out_pred.unsqueeze(1),
                label_nonsense_pred.unsqueeze(1),
                label_ileo_pred.unsqueeze(1),
                label_cls_code_pred
            ],
            dim=-1).float()

        return logit, label_pred
