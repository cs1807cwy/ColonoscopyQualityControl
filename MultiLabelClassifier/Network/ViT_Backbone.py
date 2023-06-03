import torch.nn as nn
import timm

__all__ = ['ViT_B_Extractor', 'ViT_L_Extractor', 'Swinv2_L_Window12_192_Extractor', 'ViT_L_Patch14_336_Extractor']


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


class Swinv2_L_Window12_192_Extractor(nn.Module):
    def __init__(self, pretrained: bool):
        super(Swinv2_L_Window12_192_Extractor, self).__init__()
        self.main_module = timm.create_model('swinv2_large_window12_192.ms_in22k', pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.main_module(x)


class ViT_L_Patch14_336_Extractor(nn.Module):
    def __init__(self, pretrained: bool):
        super(ViT_L_Patch14_336_Extractor, self).__init__()
        self.main_module = timm.create_model('vit_large_patch14_clip_336.openai_ft_in12k_in1k', pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.main_module(x)
