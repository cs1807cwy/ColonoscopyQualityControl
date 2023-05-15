import os

import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule
from .Net import ResNet50
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple


class SiteQualityClassifier(LightningModule):
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
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.classifier = ResNet50(num_classes=num_classes)
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

    def forward(self, x):
        return self.classifier(x)

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
            'pred_nonsense_gt_nonsense': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_nonsense_gt_outside': (torch.eq(y_hat.argmax(dim=-1), 1) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
            'pred_outside_gt_fine': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 0)).float().sum(),
            'pred_outside_gt_nonsense': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 1)).float().sum(),
            'pred_outside_gt_outside': (torch.eq(y_hat.argmax(dim=-1), 2) & torch.eq(y.argmax(dim=-1), 2)).float().sum(),
        }

        self.log_dict(confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def test_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(b1, b2))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]

    def on_test_epoch_start(self):
        os.makedirs('./ModelScript', exist_ok=True)
        self.to_torchscript('./ModelScript/model.pt', method='trace')
