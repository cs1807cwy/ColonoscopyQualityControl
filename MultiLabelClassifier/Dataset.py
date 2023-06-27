import glob
import os
import os.path as osp
import json
import random

import numpy as np
from PIL import Image
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple, Set

import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class ColonoscopyMultiLabelDataset(Dataset):
    def __init__(self,
                 # index file is under dataset root
                 data_index_file: str,
                 data_root: str = None,
                 sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = None,
                 for_validation: bool = False,
                 for_test: bool = False,  # for_test 有最高优先级
                 resize_shape: Tuple[int, int] = (224, 224),
                 center_crop_shape: Tuple[int, int] = (224, 224),
                 brightness_jitter: Union[float, Tuple[float, float]] = 0.8,
                 contrast_jitter: Union[float, Tuple[float, float]] = 0.8,
                 saturation_jitter: Union[float, Tuple[float, float]] = 0.8,
                 dry_run: bool = False,
                 **kwargs,
                 ):
        """
        data_index_file json format {
            code: [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3]
            train: {
              ileocecal:
              {
                ./ileocecal_xxxxxx.png: [0., 0., 1., 0., 0., 0., 1.]
                ...
              }
              nofeature:
              {
                ../some_dataset/nofeature_xxxxxx.png: [0., 0., 0., 0., 1., 0., 0.]
                ...
              }
              nonsense:
              {
                ../some_dataset/nonsense_xxxxxx.png: [0., 1., 0., 0., 0., 0., 0.]
                ...
              }
              outside:
              {
                C:/some_dataset/outside_xxxxxx.png: [1., 0., 0., 0., 0., 0., 0.]
                ...
              }
            }
            validation: {
              ileocecal: {...}
              nofeature: {...}
              outside: {...}
            }
        }
        """
        # 测试模式
        self.dry_run: bool = dry_run

        # 检定数据集用途
        self.for_validation: bool = for_validation
        self.for_test: bool = for_test
        if self.for_test:
            self.for_validation = True

        self.data_index_file: str = osp.abspath(data_index_file)
        self.data_root: str = osp.dirname(self.data_index_file) if data_root is None else data_root
        
        self.code_label_map: Dict[str, int] = {}

        # 数据子集索引文件内容
        self.index_content: Dict[str, Dict[str, List[float]]] = {}  # keys: code, train, validation
        self.index_length: Dict[str, int] = {}  # 数据子集索引文件长度

        with open(self.data_index_file, encoding='utf-8') as index_file:
            json_content: Dict[str, Dict] = json.load(index_file)
            self.code_label_map = {i: e for i, e in enumerate(json_content['code'])}
            if self.dry_run:
                print(f'label_code: {self.code_label_map}')

            self.index_content = json_content['validation' if self.for_validation else 'train']
            for key, val in self.index_content.items():
                self.index_content[key] = [(k, v) for k, v in val.items()]

        for key in self.index_content.keys():
            if not self.for_validation:
                random.shuffle(self.index_content[key])  # 混洗每一个用于训练的数据子集
            self.index_length[key] = len(self.index_content[key])

        # 包含所有数据子集名称的列表
        self.subset_keys: List[str] = list(self.index_content.keys())

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            # 缩放和边缘裁剪
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
        image_path, label_code = self.index_content[subset_key][inner_index]
        if not osp.isabs(image_path):
            image_path = osp.abspath(osp.join(self.data_root, image_path))
        label_code_ts: torch.Tensor = torch.from_numpy(np.array(label_code, dtype=np.float32))

        # label_code: 标签编码
        # label: 标签
        # subset_key: 采样子集
        # image_path: 图像文件路径
        if self.dry_run:
            label: List[str] = []
            for i in range(len(label_code)):
                if label_code[i] == 1.:
                    label.append(self.code_label_map[i])
                else:
                    label.append('nil')
            return label_code_ts, label, subset_key, image_path
        else:
            image: Image.Image = Image.open(image_path).convert('RGB')
            item: torch.Tensor = self.transform_validation(image) if self.for_validation else self.transform_train(image)
            return item, label_code_ts

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


class ColonoscopyMultiLabelPredictDataset(Dataset):
    def __init__(self,
                 # 数据目录
                 data_root_dir: str,
                 ext: List[str] = ('png', 'jpg'),
                 resize_shape: Tuple[int, int] = (268, 268),
                 center_crop_shape: Tuple[int, int] = (224, 224)
                 ):
        """
        Args:
            data_root_dir: str 数据目录
            ext: List[str] 有效的扩展名
            resize_shape: Tuple[高, 宽] 预处理时缩放图像的目标规格
            center_crop_shape: Tuple[高, 宽] 中心裁剪图像的目标规格，用于截去图像周围的黑边
        """

        self.data_root_dir: str = osp.abspath(data_root_dir)
        self.items: List[str] = []

        # 筛选全部具有ext指定包含后缀名的文件
        for e in ext:
            self.items += glob.glob(osp.join(self.data_root_dir, '**', f'*.{e}'), recursive=True)
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

        # 图像Tensor
        return item

    def __len__(self) -> int:
        return len(self.items)
