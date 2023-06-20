import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple

import torchvision.utils
from lightning.pytorch import LightningModule

from .Network import *
from .LossFunction import *


class MultiLabelClassifier_ViT_L_Patch16_224_Class7(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (224, 224),  # must be 224*224 as we use pretrained ViT_Patch16_224
            num_heads: int = 8,  # heads number in [1, 2, 4, 6, 8]
            attention_lambda: float = 0.3,
            num_classes: int = 7,
            thresh: float = 0.5,
            batch_size: int = 16,
            lr: float = 0.0001,
            epochs: int = 1000,
            b1: float = 0.9,
            b2: float = 0.999,
            momentum: float = 0.9,
            weight_decay: float = 0.0001,
            cls_weight: float = 4.,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.backbone = ViT_L_Patch16_224_Extractor(True)
        self.classify_head = ClassSpecificMultiHeadAttention(num_heads, attention_lambda, 1024, num_classes)  # embed dim=1024
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

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

        self.confuse_matrix: Dict = {}

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
        logit = self(image)
        # loss = F.binary_cross_entropy_with_logits(logit, label_gt, reduction='mean')
        # loss = sigmoid_focal_loss_star_jit(logit, label_gt, reduction='mean')
        loss, loss_loc, loss_cls = self._calculate_loss(logit, label_gt)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_loc', loss_loc, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_cls', loss_cls, prog_bar=True, logger=True, sync_dist=True)

        # 计算总体train_mean_acc
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = torch.eq(label_pred_tf, label_gt_tf).float().mean()
        self.log(f'train_thresh_mean_acc', mean_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def _calculate_loss(self, logit, gt):
        loss_loc = F.binary_cross_entropy_with_logits(logit, gt, reduction='mean')
        loss_cls = torch.mean(F.cross_entropy(logit[:, 3:], gt[:, 3:], reduction='none') * (1. - gt[:, 0]) * (1. - gt[:, 1]))
        return loss_loc + self.hparams.cls_weight * loss_cls, loss_loc, loss_cls

    def configure_optimizers(self):
        # return self.configure_adam_cosine()
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

    def _configure_adam_cosine(self):
        # optimizer and warmup
        backbone, classifier = [], []
        for name, param in self.named_parameters():
            if 'classify_head' in name:
                classifier.append(param)
            else:
                backbone.append(param)
        optimizer = torch.optim.AdamW(
            [
                {'params': backbone, 'lr': self.hparams.lr},
                {'params': classifier, 'lr': self.hparams.lr * 10.}
            ], betas=(self.hparams.b1, self.hparams.b2), amsgrad=True)
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
        logit = self(image)

        # 计算val_acc
        # label_pred_tf: BoolTensor[B, 7] = B * [nonsense?, outside?, ileocecal?, bbps0?, bbps1?, bbps2?, bbps3?]
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = torch.eq(label_pred_tf, label_gt_tf).float().mean()
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

    def on_test_epoch_start(self):
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

        if self.hparams.save_dir is not None:
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_outside_gt_inside'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_inside_gt_outside'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_nonsense_gt_fine'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_fine_gt_nonsense'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_ileocecal_gt_nofeature'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_nofeature_gt_ileocecal'), exist_ok=True)
            for i in range(0, 4):  # i: predict
                for j in range(0, 4):  # j: gt
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_bbps{i}_gt_bbps{j}'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_lq_gt_hq'), exist_ok=True)
            os.makedirs(os.path.join(self.hparams.save_dir, 'pred_hq_gt_lq'), exist_ok=True)

    def test_step(self, batch, batch_idx: int):
        image, label_gt = batch
        logit = self(image)

        # 计算test_acc
        # label_pred_tf: BoolTensor[B, 7] = B * [nonsense?, outside?, ileocecal?, bbps0?, bbps1?, bbps2?, bbps3?]
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = torch.eq(label_pred_tf, label_gt_tf).float().mean()
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
        if self.hparams.save_dir is not None:
            cnt = 0
            for img, lb, gt in zip(image, label_in_out_pred, label_in_out_gt):
                if lb != gt:
                    torchvision.utils.save_image(
                        img,
                        os.path.join(
                            self.hparams.save_dir,
                            'pred_outside_gt_inside' if lb else 'pred_inside_gt_outside',
                            f'batch_{batch_idx}_{cnt}.png'))
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
        if self.hparams.save_dir is not None:
            cnt = 0
            for img, lb, gt in zip(image, label_nonsense_pred, label_nonsense_gt):
                if lb != gt:
                    torchvision.utils.save_image(
                        img,
                        os.path.join(
                            self.hparams.save_dir,
                            'pred_nonsense_gt_fine' if lb else 'pred_fine_gt_nonsense',
                            f'batch_{batch_idx}_{cnt}.png'))
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
        if self.hparams.save_dir is not None:
            cnt = 0
            for img, lb, gt in zip(image, label_ileo_pred, label_ileo_gt):
                if lb != gt:
                    torchvision.utils.save_image(
                        img,
                        os.path.join(
                            self.hparams.save_dir,
                            'pred_ileocecal_gt_nofeature' if lb else 'pred_nofeature_gt_ileocecal',
                            f'batch_{batch_idx}_{cnt}.png'))
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
        if self.hparams.save_dir is not None:
            # 四分类
            cnt = 0
            for img, lb, gt in zip(image, label_cls_pred, label_cls_gt):
                if lb != gt:
                    torchvision.utils.save_image(
                        img,
                        os.path.join(
                            self.hparams.save_dir,
                            f'pred_bbps{int(lb)}_gt_bbps{int(gt)}',
                            f'batch_{batch_idx}_{cnt}.png'))
                cnt += 1
            # 二分类
            cnt = 0
            for img, lb, gt in zip(image, label_cls_pred, label_cls_gt):
                lb = int(lb)
                gt = int(gt)
                if lb in {0, 1} and gt in {2, 3}:
                    torchvision.utils.save_image(
                        img,
                        os.path.join(
                            self.hparams.save_dir,
                            f'pred_lq_gt_hq',
                            f'batch_{batch_idx}_{cnt}.png'))
                elif lb in {2, 3} and gt in {0, 1}:
                    torchvision.utils.save_image(
                        img,
                        os.path.join(
                            self.hparams.save_dir,
                            f'pred_hq_gt_lq',
                            f'batch_{batch_idx}_{cnt}.png'))
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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        image = batch  # x是图像tensor，ox是原始图像tensor
        logit = self(image)

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

        return logit, label_pred
