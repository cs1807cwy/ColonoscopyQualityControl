import torch.nn as nn
import timm

__all__ = ['VGG19']


class VGG19(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int, *args, **kwargs):
        super(VGG19, self).__init__()
        self.main_module = timm.create_model('vgg19_bn.tv_in1k', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.main_module(x)


