import torch
from lightning.pytorch import LightningModule
from typing import Tuple

from .Network import *


class ResNet50Classifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.9,
            b2: float = 0.999,
            epochs: int = 50,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.classifier = ResNet50(num_classes=num_classes, pretrained=True)
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr, betas=(b1, b2), amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]


class ResNet101Classifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.9,
            b2: float = 0.999,
            epochs: int = 50,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.classifier = ResNet101(num_classes=num_classes, pretrained=True)
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr, betas=(b1, b2), amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]


class VGG19Classifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.9,
            b2: float = 0.999,
            epochs: int = 50,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.classifier = VGG19Classifier(num_classes=num_classes, pretrained=True)
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr, betas=(b1, b2), amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]

class ViT_B_Classifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (224, 224),
            pretrained: bool = True,
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.9,
            b2: float = 0.999,
            epochs: int = 50,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.classifier = ViT_B(pretrained, num_classes)
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr, betas=(b1, b2), amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]


class ViT_L_Classifier(LightningModule):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (224, 224),
            pretrained: bool = True,
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.9,
            b2: float = 0.999,
            epochs: int = 50,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.classifier = ViT_L(pretrained, num_classes)
        self.example_input_array = [torch.zeros(batch_size, 3, input_shape[0], input_shape[1])]

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr, betas=(b1, b2), amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return [opt], [lr_scheduler]


if __name__ == '__main__':
    net = ResNet101Classifier