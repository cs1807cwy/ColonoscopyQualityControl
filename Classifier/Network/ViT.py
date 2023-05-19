import torch.nn as nn
import timm

__all__ = ['ViT_B', 'ViT_L']


class ViT_B(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int, *args, **kwargs):
        super(ViT_B, self).__init__()
        self.main_module = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.main_module(x)


class ViT_L(nn.Module):
    def __init__(
            self,
            pretrained: bool,
            num_classes: int):
        super(ViT_L, self).__init__()
        self.main_module = timm.create_model('vit_large_patch16_224.augreg_in21k', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.main_module(x)
