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
