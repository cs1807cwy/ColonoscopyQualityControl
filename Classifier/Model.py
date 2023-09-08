import os

import torch
import torch.nn.functional as F

from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union, Tuple
from collections import defaultdict

import torchvision.utils

from .BaseModel import *


class QualityClassifier(ResNet101Classifier):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 2,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'fine',
            1: 'nonsense'
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)

        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        # self.index_label: Dict[int, str] = {
        #    0: 'fine',
        #    1: 'nonsense'
        # }
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)

        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    # def on_test_epoch_start(self):
    #     os.makedirs('./ModelScript', exist_ok=True)
    #     self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')


class QualityClassifierVGG19(VGG19Classifier):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 2,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'fine',
            1: 'nonsense'
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)

        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        # self.index_label: Dict[int, str] = {
        #    0: 'fine',
        #    1: 'nonsense'
        # }
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)

        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    # def on_test_epoch_start(self):
    #     os.makedirs('./ModelScript', exist_ok=True)
    #     self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')



class CleansingClassifier(ResNet101Classifier):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'bbps0',
            1: 'bbps1',
            2: 'bbps2',
            3: 'bbps3',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_predict_epoch_start(self):
        for k1, v1 in self.index_label.items():
            os.makedirs(os.path.join(self.hparams.save_dir, f'{v1}'), exist_ok=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[List[int], List[str]]:
        x, ox = batch  # x是图像tensor，ox是原始图像tensor
        y_hat = self(x)

        for idx, sample in enumerate(zip(x, y_hat, ox)):
            img, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            torchvision.utils.save_image(
                origin_img,
                os.path.join(self.hparams.save_dir,
                             f'{self.index_label[pred_idx]}',
                             'frame_%06d.png' % (batch_idx * batch[0].size(0) + idx)))
        pred_label_codes = list(y_hat.argmax(dim=-1).cpu().numpy())
        pred_labels = [self.index_label[k] for k in pred_label_codes]
        return pred_label_codes, pred_labels


class CleansingClassifierVGG19(VGG19Classifier):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'bbps0',
            1: 'bbps1',
            2: 'bbps2',
            3: 'bbps3',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_predict_epoch_start(self):
        for k1, v1 in self.index_label.items():
            os.makedirs(os.path.join(self.hparams.save_dir, f'{v1}'), exist_ok=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[List[int], List[str]]:
        x, ox = batch  # x是图像tensor，ox是原始图像tensor
        y_hat = self(x)

        for idx, sample in enumerate(zip(x, y_hat, ox)):
            img, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            torchvision.utils.save_image(
                origin_img,
                os.path.join(self.hparams.save_dir,
                             f'{self.index_label[pred_idx]}',
                             'frame_%06d.png' % (batch_idx * batch[0].size(0) + idx)))
        pred_label_codes = list(y_hat.argmax(dim=-1).cpu().numpy())
        pred_labels = [self.index_label[k] for k in pred_label_codes]
        return pred_label_codes, pred_labels


class IleocecalClassifier(ResNet101Classifier):

    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'nofeature',
            1: 'ileocecal',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        true_positive = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[0]}']
        true_negative = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[1]}']
        precision: float = 0. if (true_positive + true_negative) == 0 else float(true_positive) / float(
            true_positive + true_negative)
        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        true_positive = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[0]}']
        true_negative = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[1]}']
        precision: float = 0. if (true_positive + true_negative) == 0 else float(true_positive) / float(
            true_positive + true_negative)
        self.log('test_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)


class IleocecalClassifierVGG19(VGG19Classifier):

    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'nofeature',
            1: 'ileocecal',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        true_positive = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[0]}']
        true_negative = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[1]}']
        precision: float = 0. if (true_positive + true_negative) == 0 else float(true_positive) / float(
            true_positive + true_negative)
        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        true_positive = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[0]}']
        true_negative = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[1]}']
        precision: float = 0. if (true_positive + true_negative) == 0 else float(true_positive) / float(
            true_positive + true_negative)
        self.log('test_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)


class PositionClassifier(ResNet101Classifier):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 2,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'inside',
            1: 'outside',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)

        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        # self.index_label: Dict[int, str] = {
        #    0: 'inside',
        #    1: 'outside'
        # }
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)

        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    # def on_test_epoch_start(self):
    #     os.makedirs('./ModelScript', exist_ok=True)
    #     self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')


class PositionClassifierVGG19(VGG19Classifier):
    def __init__(
            self,
            input_shape: Tuple[int, int] = (256, 256),
            num_classes: int = 2,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'inside',
            1: 'outside',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.,0.])
        y_hat = self(x)

        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        # self.index_label: Dict[int, str] = {
        #    0: 'inside',
        #    1: 'outside'
        # }
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)

        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    # def on_test_epoch_start(self):
    #     os.makedirs('./ModelScript', exist_ok=True)
    #     self.to_torchscript(f'./ModelScript/model_{type(self)}.pt', method='trace')


class IleocecalClassifier_ViT_B(ViT_B_Classifier):

    def __init__(
            self,
            input_shape: Tuple[int, int] = (224, 224),
            pretrained: bool = True,
            num_classes: int = 3,
            batch_size: int = 16,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            epochs: int = 50,
            save_dir: str = 'test_viz',
            **kwargs,
    ):
        super().__init__(input_shape, pretrained, num_classes, batch_size, lr, b1, b2, epochs, **kwargs)
        self.index_label: Dict[int, str] = {
            0: 'nofeature',
            1: 'ileocecal',
        }
        if 'index_label' in kwargs:
            self.index_label: Dict[int, str] = kwargs['index_label']

        self.validation_confuse_matrix = None

    def training_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # 计算train_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_confuse_matrix: defaultdict = defaultdict(int)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])
        y_hat = self(x)
        # 计算val_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.validation_confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

    def on_validation_epoch_end(self):
        true_positive = self.validation_confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[0]}']
        true_negative = self.validation_confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[1]}']
        precision: float = 0. if (true_positive + true_negative) == 0 else float(true_positive) / float(
            true_positive + true_negative)
        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.validation_confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)

    def on_test_epoch_start(self):
        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                if k1 != k2:
                    os.makedirs(os.path.join(self.hparams.save_dir, f'pred_{v1}_gt_{v2}'), exist_ok=True)
        self.confuse_matrix: defaultdict = defaultdict(int)

    def test_step(self, batch, batch_idx: int):
        x, y, ox = batch  # x是图像tensor，y是对应的标签，y形如tensor([1.,0.])，ox是原始图像tensor
        y_hat = self(x)
        # 计算test_acc
        acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        for k1, v1 in self.index_label.items():
            for k2, v2 in self.index_label.items():
                self.confuse_matrix[f'pred_{v1}_gt_{v2}'] += \
                    (torch.eq(y_hat.argmax(dim=-1), k1) & torch.eq(y.argmax(dim=-1), k2)).float().sum()

        for idx, sample in enumerate(zip(x, y, y_hat, ox)):
            img, gt_label, pred_label, origin_img = sample
            pred_idx = int(pred_label.argmax(dim=-1).cpu().numpy())
            gt_idx = int(gt_label.argmax(dim=-1).cpu().numpy())
            if gt_idx != pred_idx:
                torchvision.utils.save_image(
                    img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_augment.png'))
                torchvision.utils.save_image(
                    origin_img,
                    os.path.join(self.hparams.save_dir,
                                 f'pred_{self.index_label[pred_idx]}_gt_{self.index_label[gt_idx]}',
                                 f'batch_{batch_idx}_{idx}_origin.png'))

    def on_test_epoch_end(self):
        true_positive = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[0]}']
        true_negative = self.confuse_matrix[f'pred_{self.index_label[0]}_gt_{self.index_label[1]}']
        precision: float = 0. if (true_positive + true_negative) == 0 else float(true_positive) / float(
            true_positive + true_negative)
        self.log('test_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(self.confuse_matrix, logger=True, sync_dist=True, reduce_fx=torch.sum)
