import os

import math
import torch
import numpy as np
import torch.nn.functional as F

from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple
from collections import defaultdict

import torchvision.utils

from .Network import *
from lightning.pytorch import LightningModule


class MultiLabelClassifier_ViT_L_Patch16_224(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (224, 224),  # must be 224*224 as we use pretrained ViT_Patch16_224
            num_heads: int = 8,  # heads number in [1, 2, 4, 6, 8]
            attention_lambda: float = 0.3,
            num_classes: int = 6,
            thresh: float = 0.5,
            batch_size: int = 16,
            lr: float = 0.0001,
            momentum: float = 0.9,
            weight_decay: float = 0.0001,
            step_size: int = 5,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.backbone = ViT_L_Extractor(True)
        self.classify_head = ClassSpecificMultiHeadAttention(num_heads, attention_lambda, 1024, num_classes)  # embed dim=1024
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

        # for viz
        self.index_label: Dict[int, str] = {
            0: 'outside',
            1: 'ileocecal',
            2: 'bbps0',
            3: 'bbps1',
            4: 'bbps2',
            5: 'bbps3',
        }

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
        loss = F.binary_cross_entropy_with_logits(logit, label_gt, reduction='mean')
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        # 计算总体train_mean_acc
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = torch.eq(label_pred_tf, label_gt_tf).float().mean()
        self.log(f'train_thresh_mean_acc', mean_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=0.1)
        return [optimizer], [scheduler]

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        image, label_gt = batch
        logit = self(image)

        # 计算val_acc
        # label_pred_tf: BoolTensor[B, 6] = B * [outside?, ileocecal?, bbps0?, bbps1?, bbps2?, bbps3?]
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        mean_acc = torch.eq(label_pred_tf, label_gt_tf).float().mean()
        self.log('val_thresh_mean_acc', mean_acc, prog_bar=True, logger=True, sync_dist=True)

        # 逐标签混淆矩阵
        for k, v in self.index_label.items():
            label_pred_k = label_pred_tf[:, k]
            label_gt_k = label_gt_tf[:, k]
            self.confuse_matrix[f'label_{v}_TP'] += (label_pred_k & label_gt_k).float().sum()
            self.confuse_matrix[f'label_{v}_FP'] += (label_pred_k & ~label_gt_k).float().sum()
            self.confuse_matrix[f'label_{v}_FN'] += (~label_pred_k & label_gt_k).float().sum()
            self.confuse_matrix[f'label_{v}_TN'] += (~label_pred_k & ~label_gt_k).float().sum()

        # bbps0-1/bbps2-3二分清洁度混淆矩阵
        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 2:]
        # 清洁度label: IntTensor[B]
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度gt: IntTensor[B]
        label_cls_gt = torch.argmax(label_gt[:, 2:], dim=-1)
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 2]}_gt_{self.index_label[j + 2]}'] += \
                    (torch.eq(label_cls_pred, i) & torch.eq(label_cls_gt, j)).float().sum()

    def on_validation_epoch_end(self):
        metrics = {}
        for k, v in self.index_label.items():
            TP = self.confuse_matrix[f'label_{v}_TP']
            FP = self.confuse_matrix[f'label_{v}_FP']
            FN = self.confuse_matrix[f'label_{v}_FN']
            TN = self.confuse_matrix[f'label_{v}_TN']
            # 回盲部标签查准率
            if k == 1:
                metrics[f'label_{v}_prec'] = float(TP) / float(TP + FP) if TP + FP > 0 else 0.
            metrics[f'label_{v}_acc'] = float(TP + TN) / float(TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.

        # bbps0-1/bbps2-3二分清洁度准确率
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                cnt = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 2]}_gt_{self.index_label[j + 2]}']
                if i in {0, 1} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_TP'] += cnt
                elif i in {0, 1} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_FP'] += cnt
                elif i in {2, 3} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_FN'] += cnt
                elif i in {2, 3} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_TN'] += cnt
        TP = self.confuse_matrix['label_cleansing_low_TP'] = float(self.confuse_matrix['label_cleansing_low_TP'])
        FP = self.confuse_matrix['label_cleansing_low_FP'] = float(self.confuse_matrix['label_cleansing_low_FP'])
        FN = self.confuse_matrix['label_cleansing_low_FN'] = float(self.confuse_matrix['label_cleansing_low_FN'])
        TN = self.confuse_matrix['label_cleansing_low_TN'] = float(self.confuse_matrix['label_cleansing_low_TN'])
        metrics['label_cleansing_biclassify_acc'] = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.

        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)
        self.log_dict(metrics, logger=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        image, label_gt = batch
        logit = self(image)

        # 体内外logit: FloatTensor[B]
        in_out_logit = logit[:, 0]
        # 体内外标签: BoolTensor[B]
        # outside时为True
        label_in_out_pred = torch.ge(in_out_logit, self.hparams.thresh)
        # 体内外gt: BoolTensor[B]
        label_in_out_gt = torch.ge(label_gt[:, 0], self.hparams.thresh)
        self.confuse_matrix[f'label_{self.index_label[0]}_TP'] += (label_in_out_pred & label_in_out_gt).float().sum()
        self.confuse_matrix[f'label_{self.index_label[0]}_FP'] += (label_in_out_pred & ~label_in_out_gt).float().sum()
        self.confuse_matrix[f'label_{self.index_label[0]}_FN'] += (~label_in_out_pred & label_in_out_gt).float().sum()
        self.confuse_matrix[f'label_{self.index_label[0]}_TN'] += (~label_in_out_pred & ~label_in_out_gt).float().sum()

        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 2:]
        # 清洁度label: IntTensor[B] (取预测值最大的，但会被outside标签抑制)
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度gt: IntTensor[B]
        label_cls_gt = torch.argmax(label_gt[:, 2:], dim=-1)
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 2]}_gt_{self.index_label[j + 2]}'] += \
                    (~label_in_out_pred & torch.eq(label_cls_pred, i) & torch.eq(label_cls_gt, j)).float().sum()  # ~label_in_out_pred用于清洁度标签抑制

        # 回盲部logit: FloatTensor[B]
        ileo_logit = logit[:, 1]
        # Deprecated: 回盲部标签: BoolTensor[B] (被outside和低cls-bbps0-1标签抑制)
        # label_ileo_pred = ~label_in_out_pred & (torch.eq(label_cls_pred, 2) | torch.eq(label_cls_pred, 3)) & torch.ge(ileo_logit, self.hparams.thresh)
        # 回盲部标签: BoolTensor[B] (被outside标签抑制)
        label_ileo_pred = ~label_in_out_pred & torch.ge(ileo_logit, self.hparams.thresh)
        # 回盲部gt: BoolTensor[B]
        label_ileo_gt = torch.ge(label_gt[:, 1], self.hparams.thresh)
        self.confuse_matrix[f'label_{self.index_label[1]}_TP'] += (label_ileo_pred & label_ileo_gt).float().sum()
        self.confuse_matrix[f'label_{self.index_label[1]}_FP'] += (label_ileo_pred & ~label_ileo_gt).float().sum()
        self.confuse_matrix[f'label_{self.index_label[1]}_FN'] += (~label_ileo_pred & label_ileo_gt).float().sum()
        self.confuse_matrix[f'label_{self.index_label[1]}_TN'] += (~label_ileo_pred & ~label_ileo_gt).float().sum()

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

        metrics = {}

        # 体内外
        TP: int = self.confuse_matrix[f'label_{self.index_label[0]}_TP']
        FP: int = self.confuse_matrix[f'label_{self.index_label[0]}_FP']
        FN: int = self.confuse_matrix[f'label_{self.index_label[0]}_FN']
        TN: int = self.confuse_matrix[f'label_{self.index_label[0]}_TN']
        metrics[f'label_{self.index_label[0]}_acc'] = float(TP + TN) / float(TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.

        # 回盲部
        TP: int = self.confuse_matrix[f'label_{self.index_label[1]}_TP']
        FP: int = self.confuse_matrix[f'label_{self.index_label[1]}_FP']
        FN: int = self.confuse_matrix[f'label_{self.index_label[1]}_FN']
        TN: int = self.confuse_matrix[f'label_{self.index_label[1]}_TN']
        metrics[f'label_{self.index_label[1]}_prec'] = float(TP) / float(TP + FP) if TP + FP > 0 else 0.
        metrics[f'label_{self.index_label[1]}_acc'] = float(TP + TN) / float(TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.

        # 清洁度
        total: int = 0
        correct: int = 0
        for i in range(2, 6):  # i: predict
            for j in range(2, 6):  # j: gt
                tmp = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i]}_gt_{self.index_label[j]}']
                if i == j:
                    correct += tmp
                total += tmp
        metrics[f'label_cleansing_acc'] = float(correct) / float(total) if total > 0 else 0.

        # bbps0-1/bbps2-3二分清洁度准确率
        for i in range(0, 4):  # i: predict
            for j in range(0, 4):  # j: gt
                cnt = self.confuse_matrix[f'label_cleansing_pred_{self.index_label[i + 2]}_gt_{self.index_label[j + 2]}']
                if i in {0, 1} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_TP'] += cnt
                elif i in {0, 1} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_FP'] += cnt
                elif i in {2, 3} and j in {0, 1}:
                    self.confuse_matrix['label_cleansing_low_FN'] += cnt
                elif i in {2, 3} and j in {2, 3}:
                    self.confuse_matrix['label_cleansing_low_TN'] += cnt
        TP = self.confuse_matrix['label_cleansing_low_TP']
        FP = self.confuse_matrix['label_cleansing_low_FP']
        FN = self.confuse_matrix['label_cleansing_low_FN']
        TN = self.confuse_matrix['label_cleansing_low_TN']
        metrics['label_cleansing_biclassify_acc'] = float(TP + TN) / float(TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.

        self.log_dict(metrics, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        image = batch  # x是图像tensor，ox是原始图像tensor
        logit = self(image)

        # 体内外logit: FloatTensor[B]
        in_out_logit = logit[:, 0]
        # 体内外标签: BoolTensor[B]
        # outside时为True
        label_in_out_pred = torch.ge(in_out_logit, self.hparams.thresh)

        # 清洁度logit: FloatTensor[B, 4]
        cls_logit = logit[:, 2:]
        # 清洁度label: IntTensor[B] (取预测值最大的，但会被outside标签抑制)
        label_cls_pred = torch.argmax(cls_logit, dim=-1)
        # 清洁度label_code_pred: BoolTensor[B, 4]
        label_cls_code_pred = ~label_in_out_pred.unsqueeze(1) & torch.ge(cls_logit, torch.max(cls_logit, dim=-1)[0].unsqueeze(1))

        # 回盲部logit: FloatTensor[B]
        ileo_logit = logit[:, 1]
        # 回盲部标签: BoolTensor[B] (被outside和低cls-bbps0-1标签抑制)
        label_ileo_pred = ~label_in_out_pred & (torch.eq(label_cls_pred, 2) | torch.eq(label_cls_pred, 3)) & torch.ge(ileo_logit, self.hparams.thresh)

        # label_pred: FloatTensor[B, 6]
        label_pred = torch.cat([label_in_out_pred.unsqueeze(1), label_ileo_pred.unsqueeze(1), label_cls_code_pred], dim=-1).float()

        return logit, label_pred
