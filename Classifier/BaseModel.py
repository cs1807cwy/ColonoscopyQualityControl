import os

import torch
from lightning.pytorch import LightningModule
from .Net import ResNet50
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple


class ResNet50Classifier(LightningModule):
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
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(b1, b2))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]
