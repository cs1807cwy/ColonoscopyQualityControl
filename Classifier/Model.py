import os

import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule
from .Net import ResNet50
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple


class CQCClassifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
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
        x, y, _, _, _ = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        # 计算train_acc
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y_hat == y).item() / len(y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y, _, _, _ = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        # 计算val_acc
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y_hat == y).item() / len(y)
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx: int):
        x, y, _, _, _ = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        # 计算test_acc
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y_hat == y).item() / len(y)
        self.log('test_acc', acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(b1, b2))
        return opt

    def on_test_epoch_start(self):
        os.makedirs('./ModelScript', exist_ok=True)
        self.to_torchscript('./ModelScript/model.pt', method='trace')
