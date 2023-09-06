import torch.nn as nn
import timm

__all__ = ['ResNet50', 'ResNet101']


class ResNet50(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int, *args, **kwargs):
        super(ResNet50, self).__init__()
        self.main_module = timm.create_model('resnet50.tv_in1k', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.main_module(x)


class ResNet101(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int, *args, **kwargs):
        super(ResNet101, self).__init__()
        self.main_module = timm.create_model('resnet101.tv_in1k', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.main_module(x)
