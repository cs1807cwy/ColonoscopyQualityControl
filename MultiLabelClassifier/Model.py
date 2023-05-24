import os

import math
import torch
import torch.nn.functional as F

from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple
from collections import defaultdict

import torchvision.utils

from .Network import *
from lightning.pytorch import LightningModule


class MultilabelClassifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (224, 224),  # must be 224*224 as we use pretrained ViT_Patch16_224
            num_heads: int = 8,  # heads number in [1, 2, 4, 6, 8]
            attention_lambda: float = 0.5,
            num_classes: int = 6,
            thresh: float = 0.5,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.9,
            b2: float = 0.999,
            epochs: int = 1000,
            save_dir: str = 'test_viz',
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
        feature = feature.transpose(1, 2)
        feature = feature.reshape(b, c, h, w)
        logit = self.classify_head(feature)

        return logit

    def training_step(self, batch, batch_idx: int):
        image, label_gt = batch
        logit = self(image)
        loss = F.binary_cross_entropy_with_logits(logit, label_gt, reduction='mean')
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        # 计算train_acc
        label_pred_tf = torch.ge(logit, self.hparams.thresh)
        label_gt_tf = torch.ge(label_gt, self.hparams.thresh)
        acc = torch.eq(label_pred_tf, label_gt_tf).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr, betas=(b1, b2), amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]

    """
    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)


    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_predict_epoch_start(self):
        for k1, v1 in self.index_label.items():
            os.makedirs(os.path.join(self.hparams.save_dir, f'{v1}'), exist_ok=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[List[int], List[str]]:
        x, ox = batch  # x是图像tensor，ox是原始图像tensor
        y_hat = self(x)

        for idx, sample in enumerate(zip(x, y_hat, ox)):
            img, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            torchvision.utils.save_image(
                origin_img,
                os.path.join(self.hparams.save_dir,
                             f'{self.index_label[pred_idx]}',
                             f'frame_{batch_idx * batch[0].size(0) + idx: 0>6}.png'))
        pred_label_codes = list(y_hat.argmax(dim=-1).cpu().numpy())
        pred_labels = [self.index_label[k] for k in pred_label_codes]
        return pred_label_codes, pred_labels
    """


def TestMultiLabelClassifier():
    model = MultilabelClassifier()
    x = torch.zeros((16, 3, 224, 224))
    logit = model(x)
    print(logit.shape)
