import os
import numpy as np
import cv2
import random
from PIL import Image
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple, Set

import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from .Dataset import *


class ColonoscopyMultiLabelDataModule(LightningDataModule):
    def __init__(
            self,
            image_index_file_or_root: str,
            sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = None,
            resize_shape: Tuple[int, int] = (268, 268),
            center_crop_shape: Tuple[int, int] = (224, 224),
            brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
            contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
            saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
            batch_size: int = 16,
            num_workers: int = 0,
            dry_run: bool = False
    ):
        """
        Args:
            image_index_file_or_root: str 索引文件路径或[测试]根路径
            sample_weight: 数据子集采样率。
                如果为None则不执行采样，简单合并所有数据子集；
                如果为int型，则每个epoch对全部数据子集按固定数量采样；
                如果为float型，则每个epoch对全部数据子集按固定比例采样；
                如果为Dict型，可对每个数据子集的采样率进行规定，值类型可以是int, float，按前述规则处理，缺失键时按None规则处理
            resize_shape: Tuple[高, 宽] 预处理时缩放图像的目标规格
            center_crop_shape: Tuple[高, 宽] 中心裁剪图像的目标规格，用于截去图像周围的黑边
            brightness_jitter: 亮度随机偏移范围，值应当非负。如果为float，偏移范围为[max(0, 1 - brightness), 1 + brightness]
                如果为Tuple[float, float]，偏移范围为[min, max]
            contrast_jitter: 对比度随机偏移范围，值应当非负。如果为float，偏移范围为[max(0, 1 - contrast), 1 + contrast]
                如果为Tuple[float, float]，偏移范围为[min, max]
            saturation_jitter: 饱和度随机偏移范围，值应当非负。如果为float，偏移范围为[max(0, 1 - saturation), 1 + saturation]
                如果为Tuple[float, float]，偏移范围为[min, max]
            batch_size: 批大小
            num_workers: 加载数据的子进程数
            dry_run: 测试模式
        """

        super().__init__()
        self.image_index_file_or_root: str = image_index_file_or_root
        self.sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = sample_weight
        self.resize_shape: Tuple[int, int] = resize_shape
        self.center_crop_shape: Tuple[int, int] = center_crop_shape
        self.brightness_jitter: Union[float, Tuple[float, float]] = brightness_jitter
        self.contrast_jitter: Union[float, Tuple[float, float]] = contrast_jitter
        self.saturation_jitter: Union[float, Tuple[float, float]] = saturation_jitter
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.dry_run: bool = dry_run

        self.train_dataset: ColonoscopyMultiLabelDataset = None
        self.validation_dataset: ColonoscopyMultiLabelDataset = None
        self.test_dataset: ColonoscopyMultiLabelDataset = None
        self.predict_dataset: ColonoscopyMultiLabelPredictDataset = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # list & collect all images
            self.train_dataset = ColonoscopyMultiLabelDataset(
                self.image_index_file_or_root,
                self.sample_weight,
                False,
                False,
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
                self.dry_run
            )
            self.validation_dataset = ColonoscopyMultiLabelDataset(
                self.image_index_file_or_root,
                self.sample_weight,
                True,
                False,
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
                self.dry_run
            )
        elif stage == 'test':
            self.test_dataset = ColonoscopyMultiLabelDataset(
                self.image_index_file_or_root,
                self.sample_weight,
                True,
                True,
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
                self.dry_run
            )
        elif stage == 'predict':
            self.predict_dataset = ColonoscopyMultiLabelPredictDataset(
                self.image_index_file_or_root,
                ['png', 'jpg'],
                self.resize_shape,
                self.center_crop_shape
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    def size(self, part=None) -> Optional[int]:
        if part == 'train' or part is None:
            return len(self.train_dataset)
        elif part == 'validation':
            return len(self.validation_dataset)
        else:
            return None

    # Dict {outside: 0, ileocecal: 1, bbps0: 2, bbps1: 3, bbps2: 4, bbps3: 5}
    def get_label_code_dict(self, part=None) -> Dict[torch.Tensor, str]:
        if part == 'train' or part is None:
            return {v: k for k, v in self.train_dataset.code_label_map.items()}
        elif part == 'validation':
            return {v: k for k, v in self.validation_dataset.code_label_map.items()}
        else:
            return {}

    # Dict {0: outside, 1: ileocecal, 2: bbps0, 3: bbps1, 4: bbps2, 5: bbps3}
    def get_code_label_dict(self, part=None) -> Dict[str, torch.Tensor]:
        if part == 'train' or part is None:
            return self.train_dataset.code_label_map
        elif part == 'validation':
            return self.validation_dataset.code_label_map
        else:
            return {}


if __name__ == '__main__':
    pass
