import glob
import os.path as osp
import json
import os.path as osp
import random
from typing import Dict, List, Union, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class NerthusPredictDataset(Dataset):
    def __init__(self,
                 # 数据目录
                 image_root_dir: str,
                 ext: List[str] = ('png', 'jpg'),
                 resize_shape: Tuple[int, int] = (306, 306),
                 center_crop_shape: Tuple[int, int] = (256, 256)
                 ):
        """
        Args:
            image_dir: str 数据目录
            resize_shape: Tuple[高, 宽] 预处理时缩放图像的目标规格
            center_crop_shape: Tuple[高, 宽] 中心裁剪图像的目标规格，用于截去图像周围的黑边
            dry_run: 测试模式
        """

        self.image_root_dir: str = osp.abspath(image_root_dir)
        self.items: List[str] = []

        # 筛选全部具有ext指定包含后缀名的文件
        for e in ext:
            self.items += glob.glob(osp.join(self.image_root_dir, '**', f'*.{e}'), recursive=True)
        self.items = sorted(self.items)

        print(self.items)

        self.transform_predict = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape)
        ])

    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        image_path: str = self.items[idx]
        image: Image.Image = Image.open(image_path).convert('RGB')
        item: torch.Tensor = self.transform_predict(image)
        origin_item = transforms.ToTensor()(image)

        # 图像Tensor，原始图像Tensor，图像文件名
        return item, origin_item, image_path

    def __len__(self) -> int:
        return len(self.items)


if __name__ == '__main__':
    dataset = NerthusPredictDataset('../Datasets/UIHIMG', ['png', 'jpg'], (306, 306), (256, 256))
