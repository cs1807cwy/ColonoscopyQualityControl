import glob
import os
import os.path as osp
import json
import random

import numpy as np
import cv2
from PIL import Image
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple, Set

import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class ColonoscopySiteQualityDataset(Dataset):
    def __init__(self,
                 # Dict[数据子集名, Dict[{索引文件index|目录dir}, 路径]]
                 image_index_dir: Dict[str, Dict[str, str]],  # inner keys: index, dir
                 # Dict[数据子集名, 标签]
                 image_label: Dict[str, str],
                 sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = None,
                 for_validation: bool = False,
                 for_test: bool = False,  # for_test 有最高优先级
                 resize_shape: Tuple[int, int] = (306, 306),
                 center_crop_shape: Tuple[int, int] = (256, 256),
                 brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
                 contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
                 saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
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
            for_validation: true时作为验证集，仅当for_test为false
            for_test: true时作为测试集，最高优先级
            resize_shape: Tuple[高, 宽] 预处理时缩放图像的目标规格
            center_crop_shape: Tuple[高, 宽] 中心裁剪图像的目标规格，用于截去图像周围的黑边
            brightness_jitter: 亮度随机偏移范围，值应当非负。如果为float，偏移范围为[max(0, 1 - brightness), 1 + brightness]
                如果为Tuple[float, float]，偏移范围为[min, max]
            contrast_jitter: 对比度随机偏移范围，值应当非负。如果为float，偏移范围为[max(0, 1 - contrast), 1 + contrast]
                如果为Tuple[float, float]，偏移范围为[min, max]
            saturation_jitter: 饱和度随机偏移范围，值应当非负。如果为float，偏移范围为[max(0, 1 - saturation), 1 + saturation]
                如果为Tuple[float, float]，偏移范围为[min, max]
            dry_run: 测试模式
        """

        self.image_index_dir: Dict[str, Dict[str, str]] = image_index_dir
        self.image_label: Dict[str, str] = image_label
        self.label_code: Dict[str, torch.Tensor] = {}
        label_value_list: List[str] = list(set(self.image_label.values()))
        label_value_list = sorted(label_value_list)
        for idx, val in enumerate(label_value_list):
            onehot_code: torch.Tensor = torch.zeros(len(label_value_list))
            onehot_code[idx] = 1.
            self.label_code[val] = onehot_code
        if dry_run:
            print(f'label_code: {self.label_code}')

        self.for_validation: bool = for_validation
        self.for_test: bool = for_test
        if self.for_test:
            self.for_validation = True
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape),
            # 亮度、对比度、饱和度泛化
            transforms.ColorJitter(brightness_jitter, contrast_jitter, saturation_jitter),
            # 8方向泛化
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation([90, 90])])
        ])
        self.transform_validation = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape)
        ])

        # 测试模式
        self.dry_run: bool = dry_run

        # 包含所有数据子集名称的列表
        self.subset_keys: List[str] = list(self.image_index_dir.keys())

        # 数据子集索引文件内容
        self.index_content: Dict[str, List[str]] = {}  # inner keys: fold, train, validation
        self.index_length: Dict[str, int] = {}  # 数据子集索引文件长度
        for k in self.subset_keys:
            index_path = self.image_index_dir[k]['index']
            dir_path = self.image_index_dir[k]['dir']
            with open(index_path, encoding='utf-8') as index_file:
                index_file_content: Dict[str, Dict] = json.load(index_file)
                if self.for_validation:
                    self.index_content[k] = [osp.join(dir_path, name) for name in index_file_content['validation']]
                else:
                    self.index_content[k] = [osp.join(dir_path, name) for name in index_file_content['train']]
                    random.shuffle(self.index_content[k])  # 混洗每一个用于训练的数据子集
                self.index_length[k] = len(self.index_content[k])

        if dry_run:
            if self.for_validation:
                print('[Validation]')
            else:
                print('[Train]')
            for k in self.subset_keys:
                print(f'{k}: {len(self.index_content[k])}')

        # 映射锚点
        # Dict[数据子集的键, Tuple[数据子集内部索引, 对外索引]]
        self.index_anchor: Dict[str, Tuple[int, int]] = {}

        # 重映射
        # Dict[对外索引, Tuple[数据子集的键, 数据子集内部索引]]
        self.index_map: Dict[int, Tuple[str, int]] = {}

        # 采样数量，全部转换为整数
        self.sample_num: Dict[str, int] = {}
        self.sample_per_epoch: int = 0

        # 取数据计数
        self.count: int = 0

        if self.for_validation:
            self._validation_generate_index_map()
            self.sample_per_epoch = sum(self.index_length.values())
        else:
            for k in self.subset_keys:
                # 取完全数据子集
                if sample_weight is None:
                    self.sample_num[k] = self.index_length[k]
                # 取固定数量
                elif type(sample_weight) is int:
                    self.sample_num[k] = sample_weight
                # 按比例提取
                elif type(sample_weight) is float:
                    self.sample_num[k] = int(sample_weight * self.index_length[k])
                elif type(sample_weight) is dict:
                    # 缺失键时，取完全数据子集
                    if sample_weight.get(k) is None or type(sample_weight[k]) is None:
                        self.sample_num[k] = self.index_length[k]
                    # 取固定数量
                    elif type(sample_weight[k]) is int:
                        self.sample_num[k] = sample_weight[k]
                    # 按比例提取
                    elif type(sample_weight[k]) is float:
                        self.sample_num[k] = int(sample_weight * self.index_length[k])
                    else:
                        raise TypeError(f'sample_weight {type(sample_weight[k])} is not valid.')
                else:
                    raise TypeError(f'sample_weight {type(sample_weight)} is not valid.')

                # 累计epoch数据总量
                self.sample_per_epoch += self.sample_num[k]

                # 扩增数据量不足采样需求的数据子集
                if self.index_length[k] < self.sample_num[k]:
                    tmp_index_content = self.index_content[k]
                    while len(tmp_index_content) < self.sample_num[k]:
                        tmp_index_content += self.index_content[k]
                    self.index_content[k] = tmp_index_content
                    self.index_length[k] = len(self.index_content[k])

            # 配置映射锚点
            tmp_offset = 0
            for k in self.subset_keys:
                self.index_anchor[k] = (0, tmp_offset)
                tmp_offset += self.sample_num[k]
            self._train_generate_index_map()

    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        if not self.for_validation:
            # 计数达到epoch数据总量后重新采样
            if self.count >= self.sample_per_epoch:
                self.count = 0
                self._train_end_epoch_shuffle_update()
                self._train_generate_index_map()
            self.count += 1

        subset_key, inner_index = self.index_map[idx]
        image_path: str = self.index_content[subset_key][inner_index]
        label: str = self.image_label[subset_key]
        label_code: torch.Tensor = self.label_code[label]
        basename: str = osp.basename(image_path)

        # label_code: 标签编码
        # label: 标签
        # subset_key: 采样子集
        # image_path: 图像文件路径
        if self.dry_run:
            return label_code, label, subset_key, basename
        else:
            image: Image.Image = Image.open(image_path).convert('RGB')
            item: torch.Tensor = self.transform_validation(image) if self.for_validation else self.transform_train(
                image)
            if self.for_test:
                origin_item = transforms.ToTensor()(image)
                return item, label_code, origin_item
            else:  # validation & train
                return item, label_code

    def __len__(self) -> int:
        return self.sample_per_epoch

    # 在epoch的结束更新映射锚点和适当混洗数据集
    def _train_end_epoch_shuffle_update(self):
        for k in self.subset_keys:
            # 更新映射锚点
            inner_index, outer_index = self.index_anchor[k]
            inner_index = (inner_index + self.sample_num[k]) % self.index_length[k]
            self.index_anchor[k] = (inner_index, outer_index)

            # 混洗数据集
            # 如果下一组是最后的残余子集
            if self.index_length[k] - inner_index < self.sample_num[k]:
                head_sample_list: List[str] = self.index_content[k][:inner_index]
                random.shuffle(head_sample_list)
                self.index_content[k] = head_sample_list + self.index_content[k][inner_index:]
            # 如果下一组是新一轮子集遍历起点
            elif inner_index < self.sample_num[k]:
                tail_sample_list: List[str] = self.index_content[k][inner_index:]
                random.shuffle(tail_sample_list)
                self.index_content[k] = self.index_content[k][:inner_index] + tail_sample_list

    # 训练集生成重映射表
    def _train_generate_index_map(self):
        tmp_count: int = 0
        for k in self.subset_keys:
            # self.index_anchor: Dict[数据子集的键, Tuple[数据子集内部索引, 对外索引]]
            tmp_offset: int = self.index_anchor[k][0] % self.index_length[k]
            for _ in range(self.sample_num[k]):
                self.index_map[tmp_count] = (k, tmp_offset)
                tmp_count += 1
                tmp_offset = (tmp_offset + 1) % self.index_length[k]

    # 验证集生成重映射表
    def _validation_generate_index_map(self):
        tmp_count: int = 0
        for k in self.subset_keys:
            for idx in range(self.index_length[k]):
                self.index_map[tmp_count] = (k, idx)
                tmp_count += 1


class ColonoscopyPredictDataset(Dataset):
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

        # 图像Tensor，原始图像Tensor
        return item, origin_item

    def __len__(self) -> int:
        return len(self.items)


class SingleClassificationDataSet(Dataset):
    def __init__(self,
                 # index文件路径
                 dataset_root: str,
                 index_path: str,
                 dataset_mode: str = 'train',
                 resize_shape: Tuple[int, int] = (306, 306),
                 center_crop_shape: Tuple[int, int] = (256, 256),
                 brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
                 contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
                 saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
                 ):

        self.dataset_root: str = dataset_root
        self.dataset_mode: str = dataset_mode
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape),
            # 亮度、对比度、饱和度泛化
            transforms.ColorJitter(brightness_jitter, contrast_jitter, saturation_jitter),
            # 8方向泛化
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation([90, 90])])
        ])
        self.transform_val_test = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape)
        ])

        # index文件内容
        self.index_path: str = index_path
        self.index_all = json.load(open(self.index_path, encoding='utf-8'))
        self.index_subset = list((self.index_all[self.dataset_mode]).items())
        self.index_code: list = self.index_all['code']
        self.label_code: dict[float, torch.Tensor] = dict()

        for idx,_ in enumerate(self.index_code):
            onehot_code = torch.zeros(len(self.index_code))
            onehot_code[idx]=1.
            self.label_code[idx]=onehot_code
    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        image_path: str = osp.join(self.dataset_root, self.index_subset[idx][0])
        raw_label: float = self.index_subset[idx][1][0]
        label = self.label_code[raw_label]
        # label: 标签
        # image_path: 图像文件路径
        image: Image.Image = Image.open(image_path).convert('RGB')
        if self.dataset_mode == 'train':
            item: torch.Tensor = self.transform_train(image)
        else:  # validation & test
            item: torch.Tensor = self.transform_val_test(image)
        if self.dataset_mode == 'test':
            origin_item = transforms.ToTensor()(image)
            return item, label, origin_item
        else:  # validation & train
            return item, label

    def __len__(self) -> int:
        return len(self.index_subset)


class MultiClassificationDataSet(Dataset):
    def __init__(self,
                 # index文件路径
                 dataset_root: str,
                 index_path: str,
                 dataset_mode: str = 'train',
                 resize_shape: Tuple[int, int] = (306, 306),
                 center_crop_shape: Tuple[int, int] = (256, 256),
                 brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
                 contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
                 saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
                 ):

        self.dataset_root: str = dataset_root
        self.dataset_mode: str = dataset_mode
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape),
            # 亮度、对比度、饱和度泛化
            transforms.ColorJitter(brightness_jitter, contrast_jitter, saturation_jitter),
            # 8方向泛化
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation([90, 90])])
        ])
        self.transform_validation_test = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和截去黑边
            transforms.Resize(resize_shape, antialias=True),
            transforms.CenterCrop(center_crop_shape)
        ])

        # index文件内容
        self.index_path: str = index_path
        self.index_all = json.load(open(self.index_path, encoding='utf-8'))
        self.index_subset = list((self.index_all[self.dataset_mode]).items())
        self.index_code: list = self.index_all['code']
        self.label_code: dict[float, torch.Tensor] = dict()

        for idx,_ in enumerate(self.index_code):
            onehot_code = torch.zeros(len(self.index_code))
            onehot_code[idx]=1.
            self.label_code[idx]=onehot_code
    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        image_path: str = osp.join(self.dataset_root, self.index_subset[idx][0])
        raw_label: float = self.index_subset[idx][1]
        label = torch.Tensor(raw_label)
        # label: 标签
        # image_path: 图像文件路径
        image: Image.Image = Image.open(image_path).convert('RGB')
        if self.dataset_mode == 'train':
            item: torch.Tensor = self.transform_train(image)
        else:  # validation & test
            item: torch.Tensor = self.transform_validation_test(image)
        if self.dataset_mode == 'test':
            origin_item = transforms.ToTensor()(image)
            return item, label, origin_item
        else:  # validation & train
            return item, label

    def __len__(self) -> int:
        return len(self.index_subset)
