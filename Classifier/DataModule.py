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


class ColonoscopySiteQualityDataModule(LightningDataModule):
    """
    # 获取类别编码的方法调用\n
    [One-Hot Label Code for Reference]\n
    cqc_data_module = ColonoscopySiteQualityDataModule(...)\n
    # 调用此数据模型的如下方法来获取标签和编码的对应关系，标签按字典顺序依次编码\n
    cqc_data_module.get_label_code_dict()  # 获取label到code的映射\n
    label to code {'fine': tensor([1., 0., 0.]), 'nonsense': tensor([0., 1., 0.]), 'outside': tensor([0., 0., 1.])}\n
    cqc_data_module.get_code_label_dict()  # 获取code到label的映射\n
    [code to label] {tensor([1., 0., 0.]): 'fine', tensor([0., 1., 0.]): 'nonsense', tensor([0., 0., 1.]): 'outside'}
    """

    def __init__(
            self,
            # Dict[数据子集名, Dict[{索引文件index|目录dir}, 路径]]
            image_index_dir: Union[str, Dict[str, Dict[str, str]]],  # inner keys: index, dir
            # Dict[数据子集名, 标签]
            image_label: Dict[str, str] = None,
            sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = None,
            resize_shape: Tuple[int, int] = (306, 306),
            center_crop_shape: Tuple[int, int] = (256, 256),
            brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
            contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
            saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
            batch_size: int = 16,
            num_workers: int = 0,
            dry_run: bool = False
    ):
        """
        Args:
            image_index_dir: Dict[数据子集名, Dict[{索引文件index|目录dir}, 路径]]
            image_label: Dict[数据子集名, 标签] 每个数据子集有且仅有一个标签
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
        self.image_index_dir: Union[str, Dict[str, Dict[str, str]]] = image_index_dir
        self.image_label: Dict[str, str] = image_label
        self.sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = sample_weight
        self.resize_shape: Tuple[int, int] = resize_shape
        self.center_crop_shape: Tuple[int, int] = center_crop_shape
        self.brightness_jitter: Union[float, Tuple[float, float]] = brightness_jitter
        self.contrast_jitter: Union[float, Tuple[float, float]] = contrast_jitter
        self.saturation_jitter: Union[float, Tuple[float, float]] = saturation_jitter
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.dry_run: bool = dry_run

        self.train_dataset: ColonoscopySiteQualityDataset = None
        self.validation_dataset: ColonoscopySiteQualityDataset = None
        self.test_dataset: ColonoscopySiteQualityDataset = None
        self.predict_dataset: ColonoscopyPredictDataset = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # list & collect all images
            self.train_dataset = ColonoscopySiteQualityDataset(
                self.image_index_dir,
                self.image_label,
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
            self.validation_dataset = ColonoscopySiteQualityDataset(
                self.image_index_dir,
                self.image_label,
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
            self.test_dataset = ColonoscopySiteQualityDataset(
                self.image_index_dir,
                self.image_label,
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
            self.predict_dataset = ColonoscopyPredictDataset(
                self.image_index_dir,
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

    def get_label_code_dict(self, part=None) -> Dict[str, torch.Tensor]:
        if part == 'train' or part is None:
            return self.train_dataset.label_code
        elif part == 'validation':
            return self.validation_dataset.label_code
        else:
            return {}

    def get_code_label_dict(self, part=None) -> Dict[torch.Tensor, str]:
        if part == 'train' or part is None:
            return {v: k for k, v in self.train_dataset.label_code.items()}
        elif part == 'validation':
            return {v: k for k, v in self.validation_dataset.label_code.items()}
        else:
            return {}


class SingleClassificationDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_root: str,
            index_file_path: str = None,
            resize_shape: Tuple[int, int] = (306, 306),
            center_crop_shape: Tuple[int, int] = (256, 256),
            brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
            contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
            saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
            batch_size: int = 16,
            num_workers: int = 0,
            ext: List[str] = ('png', 'jpg'),
    ):
        """
        Args:
            dataset_root: 数据集根目录
            index_file_path: str 索引文件路径
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
        """

        super().__init__()
        self.dataset_root: str = dataset_root
        self.ext: List[str] = ext
        self.index_file_path: str = index_file_path
        self.resize_shape: Tuple[int, int] = resize_shape
        self.center_crop_shape: Tuple[int, int] = center_crop_shape
        self.brightness_jitter: Union[float, Tuple[float, float]] = brightness_jitter
        self.contrast_jitter: Union[float, Tuple[float, float]] = contrast_jitter
        self.saturation_jitter: Union[float, Tuple[float, float]] = saturation_jitter
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_dataset: SingleClassificationDataSet = None
        self.validation_dataset: SingleClassificationDataSet = None
        self.test_dataset: SingleClassificationDataSet = None
        self.predict_dataset: ColonoscopyPredictDataset = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # list & collect all images
            self.train_dataset = SingleClassificationDataSet(
                self.dataset_root,
                self.index_file_path,
                'train',
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
            )
            self.validation_dataset = SingleClassificationDataSet(
                self.dataset_root,
                self.index_file_path,
                'validation',
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
            )
        elif stage == 'test':
            self.test_dataset = SingleClassificationDataSet(
                self.dataset_root,
                self.index_file_path,
                'test',
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
            )
        elif stage == 'predict':
            self.predict_dataset = ColonoscopyPredictDataset(
                self.dataset_root,
                self.ext,
                self.resize_shape,
                self.center_crop_shape,
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


class MultiClassificationDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_root: str,
            index_file_path: str = None,
            resize_shape: Tuple[int, int] = (306, 306),
            center_crop_shape: Tuple[int, int] = (256, 256),
            brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
            contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
            saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
            batch_size: int = 16,
            num_workers: int = 0,
            ext: List[str] = ('png', 'jpg'),
    ):
        """
        Args:
            dataset_root: 数据集根目录
            index_file_path: str 索引文件路径
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
        """

        super().__init__()
        self.dataset_root: str = dataset_root
        self.ext: List[str] = ext
        self.index_file_path: str = index_file_path
        self.resize_shape: Tuple[int, int] = resize_shape
        self.center_crop_shape: Tuple[int, int] = center_crop_shape
        self.brightness_jitter: Union[float, Tuple[float, float]] = brightness_jitter
        self.contrast_jitter: Union[float, Tuple[float, float]] = contrast_jitter
        self.saturation_jitter: Union[float, Tuple[float, float]] = saturation_jitter
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_dataset: MultiClassificationDataSet = None
        self.validation_dataset: MultiClassificationDataSet = None
        self.test_dataset: MultiClassificationDataSet = None
        self.predict_dataset: ColonoscopyPredictDataset = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # list & collect all images
            self.train_dataset = MultiClassificationDataSet(
                self.dataset_root,
                self.index_file_path,
                'train',
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
            )
            self.validation_dataset = MultiClassificationDataSet(
                self.dataset_root,
                self.index_file_path,
                'validation',
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
            )
        elif stage == 'test':
            self.test_dataset = MultiClassificationDataSet(
                self.dataset_root,
                self.index_file_path,
                'test',
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
            )
        elif stage == 'predict':
            self.predict_dataset = ColonoscopyPredictDataset(
                self.dataset_root,
                self.ext,
                self.resize_shape,
                self.center_crop_shape,
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

if __name__ == '__main__':
    pass
