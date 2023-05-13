from typing import Tuple, Dict, Union

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from Classifier.DataModule import ColonoscopySiteQualityDataModule as CQCDataModule
from Classifier.Model import SiteQualityClassifier

# hparams for Model
input_shape: Tuple[int, int] = (256, 256)
num_classes: int = 3
lr: float = 1e-4
b1: float = 0.5
b2: float = 0.999

batch_size: int = 16

# hparams for DataModule
root_dir: str = '/mnt/data4/cwy/Datasets'
image_index_dir: Dict[str, Dict[str, str]] = {
    'UIHIMG-ileocecal': {'index': f'{root_dir}/KIndex/UIHIMG/ileocecal/fold0.json',
                         'dir': f'{root_dir}/UIHIMG/ileocecal'},
    'UIHIMG-nofeature': {'index': f'{root_dir}/KIndex/UIHIMG/nofeature/fold0.json',
                         'dir': f'{root_dir}/UIHIMG/nofeature'},
    'UIHIMG-nonsense': {'index': f'{root_dir}/KIndex/UIHIMG/nonsense/fold0.json',
                        'dir': f'{root_dir}/UIHIMG/nonsense'},
    'UIHIMG-outside': {'index': f'{root_dir}/KIndex/UIHIMG/outside/fold0.json',
                       'dir': f'{root_dir}/UIHIMG/outside'},
    'Nerthus-0': {'index': f'{root_dir}/KIndex/Nerthus/0/fold0.json',
                  'dir': f'{root_dir}/Nerthus/0'},
    'Nerthus-1': {'index': f'{root_dir}/KIndex/Nerthus/1/fold0.json',
                  'dir': f'{root_dir}/Nerthus/1'},
    'Nerthus-2': {'index': f'{root_dir}/KIndex/Nerthus/2/fold0.json',
                  'dir': f'{root_dir}/Nerthus/2'},
    'Nerthus-3': {'index': f'{root_dir}/KIndex/Nerthus/3/fold0.json',
                  'dir': f'{root_dir}/Nerthus/3'},
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
resize_shape: Tuple[int, int] = (306, 306)
center_crop_shape: Tuple[int, int] = (256, 256)
brightness_jitter: Union[float, Tuple[float, float]] = 0.8
contrast_jitter: Union[float, Tuple[float, float]] = 0.8
saturation_jitter: Union[float, Tuple[float, float]] = 0.8
num_workers: int = 28

# hparams for Trainer
max_epochs: int = 50
train_save_point_epochs: int = 1
device = [0, 1]


def init_data_module():
    data_module = CQCDataModule(
        image_index_dir,
        image_label,
        sample_weight,
        resize_shape,
        center_crop_shape,
        brightness_jitter,
        contrast_jitter,
        saturation_jitter,
        batch_size,
        num_workers
    )
    return data_module


def init_trainer():
    trainer = CQCTrainer()
    return trainer


class CQCTrainer(Trainer):
    def __init__(self):
        tensorboard = TensorBoardLogger(save_dir='Experiment', name='CQCClassifier_logs',
                                        version='tensorboard_train_val')
        csv = CSVLogger(save_dir='Experiment', name='CQCClassifier_logs', version='csv_train_val')
        checkpoint_callback_regular = ModelCheckpoint(
            save_last=True,
            every_n_epochs=train_save_point_epochs,
            filename='CQCClassifier_{epoch:02d}',
            save_top_k=-1
        )
        checkpoint_callback_best_val_acc = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            filename='CQCClassifier_best_val_acc_{epoch:02d}_{val_acc:.4f}_{val_loss:.4f}'
        )
        super().__init__(logger=[tensorboard, csv],
                         default_root_dir='Experiment/CQCClassifier_logs',
                         accelerator='gpu',
                         devices=device,
                         max_epochs=max_epochs,
                         callbacks=[TQDMProgressBar(refresh_rate=20),
                                    checkpoint_callback_regular,
                                    checkpoint_callback_best_val_acc],
                         strategy='ddp',  # use build-in default DDPStrategy, it casts FLAG find_unused_parameters=True
                         check_val_every_n_epoch=1,
                         log_every_n_steps=100)


def train():
    data_module = init_data_module()
    model = SiteQualityClassifier(input_shape, num_classes, batch_size, lr, b1, b2)
    trainer = init_trainer()
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()
