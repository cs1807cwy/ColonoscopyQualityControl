import os

import torch
import torch.nn.functional as F
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple

from .BaseModel import ResNet50Classifier


class SiteQualityClassifier(ResNet50Classifier):
    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        # fine nonsense outside，合并前两类为一类进行统计
        in_out_acc = (torch.ne(y_hat.argmax(dim=-1), 2) == torch.ne(y.argmax(dim=-1), 2)).float().mean()
        self.log('train_in_out_acc', in_out_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算val_acc
        # fine nonsense outside，合并前两类为一类进行统计
        in_out_acc = (torch.ne(y_hat.argmax(dim=-1), 2) == torch.ne(y.argmax(dim=-1), 2)).float().mean()
        self.log('val_in_out_acc', in_out_acc, prog_bar=True, logger=True, sync_dist=True)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        confuse_matrix: Dict[str, int] = {
            'pred_fine_gt_fine': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_fine_gt_nonsense': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_fine_gt_outside': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_nonsense_gt_fine': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_nonsense_gt_nonsense': (
                    torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_nonsense_gt_outside': (
                    torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_outside_gt_fine': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_outside_gt_nonsense': (
                    torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_outside_gt_outside': (
                    torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
        }

        self.log_dict(confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def test_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

    def on_test_epoch_start(self):
        os.makedirs('./ModelScript', exist_ok=True)
        self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')


class IleocecalClassifier(ResNet50Classifier):

    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.val_confuse_matrix = None
        self.test_confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)
        self.val_confuse_matrix: Dict[str, int] = {
            'pred_ileocecal_gt_ileocecal': (
                    torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_ileocecal_gt_nofeature': (
                    torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_nofeature_gt_ileocecal': (
                    torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_nofeature_gt_nofeature': (
                    torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
        }
        self.log_dict(self.val_confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def test_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)
        self.test_confuse_matrix: Dict[str, int] = {
            'pred_ileocecal_gt_ileocecal': (
                    torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_ileocecal_gt_nofeature': (
                    torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_nofeature_gt_ileocecal': (
                    torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_nofeature_gt_nofeature': (
                    torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
        }
        self.log_dict(self.test_confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_validation_epoch_end(self):
        precision = self.val_confuse_matrix['pred_ileocecal_gt_ileocecal'] / \
                    (self.val_confuse_matrix['pred_ileocecal_gt_ileocecal']
                     + self.val_confuse_matrix['pred_ileocecal_gt_nofeature'])
        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True)

    def on_test_epoch_end(self):
        precision = self.test_confuse_matrix['pred_ileocecal_gt_ileocecal'] / \
                    (self.test_confuse_matrix['pred_ileocecal_gt_ileocecal']
                     + self.test_confuse_matrix['pred_ileocecal_gt_nofeature'])
        self.log('test_precision', precision, prog_bar=True, logger=True, sync_dist=True)

    def on_test_epoch_start(self):
        os.makedirs('./ModelScript', exist_ok=True)
        self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')


class CleansingClassifier(ResNet50Classifier):
    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        confuse_matrix: Dict[str, int] = {
            'pred_bbps0_gt_bbps0': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_bbps0_gt_bbps1': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_bbps0_gt_bbps2': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_bbps0_gt_bbps3': (torch.eq(y_hat.argmax(dim=-1), 0) & torch.eq(y.argmax(dim=-1), 3)).float().sum(),
            'pred_bbps1_gt_bbps0': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_bbps1_gt_bbps1': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_bbps1_gt_bbps2': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_bbps1_gt_bbps3': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 3)).float().sum(),
            'pred_bbps2_gt_bbps0': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_bbps2_gt_bbps1': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_bbps2_gt_bbps2': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_bbps2_gt_bbps3': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 3)).float().sum(),
            'pred_bbps3_gt_bbps0': (torch.eq(y_hat.argmax(dim=-1), 3) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_bbps3_gt_bbps1': (torch.eq(y_hat.argmax(dim=-1), 3) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_bbps3_gt_bbps2': (torch.eq(y_hat.argmax(dim=-1), 3) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_bbps3_gt_bbps3': (torch.eq(y_hat.argmax(dim=-1), 3) & torch.eq(y.argmax(dim=-1), 3)).float().sum()

        }

        self.log_dict(confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def test_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

    def on_test_epoch_start(self):
        os.makedirs('./ModelScript', exist_ok=True)
        self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')
