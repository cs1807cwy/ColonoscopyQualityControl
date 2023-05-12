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
from Dataset import ColonoscopySiteQualityDataset


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
            image_index_dir: Dict[str, Dict[str, str]],  # inner keys: index, dir
            # Dict[数据子集名, 标签]
            image_label: Dict[str, str],
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
        self.image_index_dir: Dict[str, Dict[str, str]] = image_index_dir
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

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # list & collect all images
            self.train_dataset = ColonoscopySiteQualityDataset(
                self.image_index_dir,
                self.image_label,
                self.sample_weight,
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
                self.resize_shape,
                self.center_crop_shape,
                self.brightness_jitter,
                self.contrast_jitter,
                self.saturation_jitter,
                self.dry_run
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

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


def TestColonoscopySiteQualityDataModule():
    # Dict[数据子集名, Dict[{索引文件index|目录dir}, 路径]]
    image_index_dir: Dict[str, Dict[str, str]] = {
        'UIHIMG-ileocecal': {'index': '../Datasets/KIndex/UIHIMG/ileocecal/fold0.json',
                             'dir': '../Datasets/UIHIMG/ileocecal'},
        'UIHIMG-nofeature': {'index': '../Datasets/KIndex/UIHIMG/nofeature/fold0.json',
                             'dir': '../Datasets/UIHIMG/nofeature'},
        'UIHIMG-nonsense': {'index': '../Datasets/KIndex/UIHIMG/nonsense/fold0.json',
                            'dir': '../Datasets/UIHIMG/nonsense'},
        'UIHIMG-outside': {'index': '../Datasets/KIndex/UIHIMG/outside/fold0.json',
                           'dir': '../Datasets/UIHIMG/outside'},
        'Nerthus-0': {'index': '../Datasets/KIndex/Nerthus/0/fold0.json',
                      'dir': '../Datasets/Nerthus/0'},
        'Nerthus-1': {'index': '../Datasets/KIndex/Nerthus/1/fold0.json',
                      'dir': '../Datasets/Nerthus/1'},
        'Nerthus-2': {'index': '../Datasets/KIndex/Nerthus/2/fold0.json',
                      'dir': '../Datasets/Nerthus/2'},
        'Nerthus-3': {'index': '../Datasets/KIndex/Nerthus/3/fold0.json',
                      'dir': '../Datasets/Nerthus/3'},
    }
    # Dict[数据子集名, 标签]
    image_label: Dict[str, str] = {
        'UIHIMG-ileocecal': 'fine',
        'UIHIMG-nofeature': 'fine',
        'UIHIMG-nonsense': 'nonsense',
        'UIHIMG-outside': 'outside',
        'Nerthus-0': 'fine',
        'Nerthus-1': 'fine',
        'Nerthus-2': 'fine',
        'Nerthus-3': 'fine',
    }
    sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = {
        'UIHIMG-ileocecal': 1666,
        'UIHIMG-nofeature': 1666,
        'UIHIMG-nonsense': 5000,
        'UIHIMG-outside': 5000,
        'Nerthus-0': 417,
        'Nerthus-1': 417,
        'Nerthus-2': 417,
        'Nerthus-3': 417,
    }
    # sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = {
    #     'UIHIMG-ileocecal': 16,
    #     'UIHIMG-nofeature': 16,
    #     'UIHIMG-nonsense': 50,
    #     'UIHIMG-outside': 50,
    #     'Nerthus-0': 4,
    #     'Nerthus-1': 4,
    #     'Nerthus-2': 4,
    #     'Nerthus-3': 4,
    # }

    resize_shape: Tuple[int, int] = (306, 306)
    center_crop_shape: Tuple[int, int] = (256, 256)
    brightness_jitter: Union[float, Tuple[float, float]] = 0.8
    contrast_jitter: Union[float, Tuple[float, float]] = 0.8
    saturation_jitter: Union[float, Tuple[float, float]] = 0.8
    batch_size: int = 16
    num_workers: int = 0

    cqc_data_module: ColonoscopySiteQualityDataModule = ColonoscopySiteQualityDataModule(
        image_index_dir,
        image_label,
        sample_weight,
        resize_shape,
        center_crop_shape,
        brightness_jitter,
        contrast_jitter,
        saturation_jitter,
        batch_size,
        num_workers,
        True
    )
    cqc_data_module.setup('fit')

    # 统计表
    # Dict[
    #   标签,
    #   Dict{
    #       sample_count: 采样标签计数,
    #       item_count: 不重复的项目计数,
    #       content: Dict[
    #           原始标签,
    #           Dict{
    #               sample_count: 采样原始标签计数,
    #               item_count: 不重复的项目计数,
    #               content: Dict[图像文件名, 计数]
    #           }
    #       ]
    #   }
    # ]
    item_counter: Dict[str, Dict[str, Optional[int, Dict[str, Dict[str, Optional[int, Dict[str, int]]]]]]] = {}

    train_dataloader = cqc_data_module.train_dataloader()

    from tqdm import tqdm
    epochs: int = 21
    samples: int = epochs * cqc_data_module.size('train')
    with tqdm(total=samples) as pbar:
        pbar.set_description('Processing')
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                # 依次为
                # item: 图像Tensor
                # label: 包装后的3类标签{outside|nonsense|fine}
                # origin_label: 原始标签{UIHIMG-ileocecal|Nerthus-0|...}
                # basename: 图像文件名
                item, label_code, label, origin_label, basename = batch
                # print(f'\tBatch {batch_idx}:' + str({'label': label, 'basename': basename}))
                for lc, lb, olb, bn in zip(label_code, label, origin_label, basename):
                    if item_counter.get(lb) is None:
                        item_counter[lb] = {
                            'sample_count': 1, 'item_count': 0,
                            'code': lc.tolist(),
                            'content': {olb: {'sample_count': 1, 'item_count': 0, 'content': {bn: 1}}}
                        }
                    elif item_counter[lb]['content'].get(olb) is None:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb] = {'sample_count': 1, 'item_count': 0, 'content': {bn: 1}}
                    elif item_counter[lb]['content'][olb]['content'].get(bn) is None:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['content'][bn] = 1
                    else:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['content'][bn] += 1
                pbar.update(len(item))

    # 计算不重复项数
    for lb in item_counter:
        for olb in item_counter[lb]['content']:
            itc = len(item_counter[lb]['content'][olb]['content'])
            item_counter[lb]['content'][olb]['item_count'] = itc
            item_counter[lb]['item_count'] += itc

    # 获取类别编码的方法调用
    print('[One-Hot Label Code for Reference]')
    print('[label to code]', cqc_data_module.get_label_code_dict('train'))
    print('[code to label]', cqc_data_module.get_code_label_dict('train'))

    import json
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/count_log.json', 'w') as count_file:
        json.dump(item_counter, count_file, indent=2)

    import matplotlib.pyplot as plt
    # 采样总数
    x_label: List[str] = []
    y_count: List[int] = []
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['sample_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['sample_count'])
    plt.figure()
    plt.bar()


    # 覆盖总数
    x_label: List[str] = []
    y_count: List[int] = []
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['item_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['item_count'])

    # 采样频数分布直方图
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['item_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['item_count'])


if __name__ == '__main__':
    TestColonoscopySiteQualityDataModule()
