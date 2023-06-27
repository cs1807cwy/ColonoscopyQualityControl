import torch.nn as nn
import timm

__all__ = ['ViT_L_Patch16_224_Extractor']

"""
Original ViT-L Implementation at https://github.com/google-research/vision_transformer
See Original Work "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  at https://iclr.cc/virtual/2021/oral/3458
"""


class ViT_L_Patch16_224_Extractor(nn.Module):
    def __init__(self, pretrained: bool):
        super(ViT_L_Patch16_224_Extractor, self).__init__()
        self.main_module = timm.create_model('vit_large_patch16_224.augreg_in21k', pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.main_module(x)
