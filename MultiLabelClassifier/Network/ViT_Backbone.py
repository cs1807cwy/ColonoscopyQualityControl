import torch.nn as nn
import timm

__all__ = ['ViT_B_Extractor', 'ViT_L_Extractor']


class ViT_B_Extractor(nn.Module):
    def __init__(self, pretrained: bool):
        super(ViT_B_Extractor, self).__init__()
        self.main_module = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.main_module(x)


class ViT_L_Extractor(nn.Module):
    def __init__(self, pretrained: bool):
        super(ViT_L_Extractor, self).__init__()
        self.main_module = timm.create_model('vit_large_patch16_224.augreg_in21k', pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.main_module(x)
