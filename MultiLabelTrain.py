import warnings

import torch
import lightning

import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from MultiLabelClassifier import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

# hparams for Trainer
accelerator = 'gpu'
strategy = 'ddp'
devices = [2, 3]
max_epochs = 1000
check_val_every_n_epoch = 1
log_every_n_steps = 10
default_root_dir = 'Experiment/Releasev1_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch1000'
logger = [
    TensorBoardLogger(
        save_dir='Experiment',
        name='Releasev1_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch1000',
        version='tensorboard_train_val'
    ),
    CSVLogger(
        save_dir='Experiment',
        name='Releasev1_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch1000',
        version='csv_train_val'
    )
]
callbacks = [
    TQDMProgressBar(refresh_rate=20),
    ModelCheckpoint(
        save_last=True,
        monitor='epoch',
        mode='max',
        every_n_epochs=max(1, max_epochs // 10),
        filename='WMuL_{epoch:03d}',
        save_top_k=10
    ),
    ModelCheckpoint(
        monitor='val_thresh_mean_acc',
        mode='max',
        filename='WMuL_best_mAcc_{epoch:03d}_{val_thresh_mean_acc:.4f}'
    ),
    ModelCheckpoint(
        monitor='label_outside_acc',
        mode='max',
        filename='WMuL_best_ioAcc_{epoch:03d}_{label_outside_acc:.4f}'
    ),
    ModelCheckpoint(
        monitor='label_nonsense_acc',
        mode='max',
        filename='WMuL_best_nsAcc_{epoch:03d}_{label_nonsense_acc:.4f}'
    ),
    ModelCheckpoint(
        monitor='label_ileocecal_acc',
        mode='max',
        filename='WMuL_best_ileoAcc_{epoch:03d}_{label_ileocecal_acc:.4f}'
    ),
    ModelCheckpoint(
        monitor='label_ileocecal_prec',
        mode='max',
        filename='WMuL_best_ileoPrec_{epoch:03d}_{label_ileocecal_prec:.4f}'
    ),
    ModelCheckpoint(
        monitor='label_cleansing_acc',
        mode='max',
        filename='WMuL_best_cls4Acc_{epoch:03d}_{label_cleansing_acc:.4f}'
    ),
    ModelCheckpoint(
        monitor='label_cleansing_biclassify_acc',
        mode='max',
        filename='WMuL_best_cls2Acc_{epoch:03d}_{label_cleansing_biclassify_acc:.4f}'
    ),
]

# hparams for DataModule
data_class_path = ColonoscopyMultiLabelDataModule
data_index_file = '/mnt/data/cwy/Datasets/UIHNJMuL/folds/fold0.json'
data_root = '/mnt/data/cwy/Datasets/UIHNJMuL'
sample_weight = {
    'ileocecal': 4800,
    'nofeature': 4800,
    'nonsense': 96,
    'outside': 96,
}
resize_shape = (224, 224)
center_crop_shape = (224, 224)
brightness_jitter = 0.8
contrast_jitter = 0.8
saturation_jitter = 0.8
batch_size = 48
num_workers = 8
dry_run = False

# hparams for Module
model_class_path = MultiLabelClassifier_ViT_L_Patch16_224_Class7
input_shape = (224, 224)  # 主干网络固定输入规格为(224, 224)，请勿修改！
num_heads = 8
attention_lambda = 0.3
num_classes = 7
thresh = 0.5
# batch_size
lr = 0.0001
epochs = max_epochs
momentum = 0.9
weight_decay = 0.0001
cls_weight = 0.2

# global settings
seed_everything = 0
ckpt_path = None


class MultiLabelClassifyLauncher:
    def __init__(self, args):
        # hparams for Trainer
        self.accelerator = accelerator
        self.strategy = strategy
        self.devices = devices
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.log_every_n_steps = log_every_n_steps
        self.default_root_dir = default_root_dir
        self.logger = logger
        self.callbacks = callbacks

        # hparams for DataModule
        self.data_class_path = data_class_path
        self.data_index_file = data_index_file
        self.data_root = data_root
        self.sample_weight = sample_weight
        self.resize_shape = resize_shape
        self.center_crop_shape = center_crop_shape
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dry_run = dry_run

        # hparams for Module
        self.model_class_path = model_class_path
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.attention_lambda = attention_lambda
        self.num_classes = num_classes
        self.thresh = thresh
        # batch_size
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cls_weight = cls_weight
        self.viz_save_dir = args.viz_save_dir

        # global settings
        self.seed_everything = seed_everything
        self.ckpt_path = ckpt_path

        if self.seed_everything is not None:
            lightning.seed_everything(self.seed_everything)

        # custom settings
        self.model_save_path = args.model_save_path
        self.compile_model = args.compile_model

    def get_trainer(self) -> Trainer:
        trainer = Trainer(
            accelerator=self.accelerator,
            strategy=self.strategy,
            devices=self.devices,
            logger=self.logger,
            callbacks=self.callbacks,
            max_epochs=self.max_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            log_every_n_steps=self.log_every_n_steps,
            default_root_dir=self.default_root_dir
        )
        return trainer

    def get_data(self) -> LightningDataModule:
        data = self.data_class_path(
            data_index_file=self.data_index_file,
            data_root=self.data_root,
            sample_weight=self.sample_weight,
            resize_shape=self.resize_shape,
            center_crop_shape=self.center_crop_shape,
            brightness_jitter=self.brightness_jitter,
            contrast_jitter=self.contrast_jitter,
            saturation_jitter=self.saturation_jitter,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            dry_run=self.dry_run
        )
        return data

    def get_model(self) -> LightningModule:
        model = model_class_path(
            input_shape=self.input_shape,
            num_heads=self.num_heads,
            attention_lambda=self.attention_lambda,
            num_classes=self.num_classes,
            thresh=self.thresh,
            batch_size=self.batch_size,
            lr=self.lr,
            epochs=self.epochs,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            cls_weight=self.cls_weight,
            save_dir=self.viz_save_dir
        )
        return model

    def launch(self, stage):
        model = self.get_model()
        if self.compile_model:
            model = torch.compile(model, mode='default')  # mode=['default', 'reduce-overhead', 'max-autotune']
        data = self.get_data()
        trainer = self.get_trainer()

        if stage == 'fit':
            trainer.fit(
                model=model,
                datamodule=data,
                ckpt_path=self.ckpt_path
            )
        elif stage == 'validate':
            trainer.validate(
                model=model,
                datamodule=data,
                ckpt_path=self.ckpt_path
            )
        elif stage == 'test':
            trainer.test(
                model=model,
                datamodule=data,
                ckpt_path=self.ckpt_path
            )
        elif stage == 'predict':
            trainer.predict(
                model=model,
                datamodule=data,
                ckpt_path=self.ckpt_path
            )
        elif stage == 'export_model':
            if self.model_save_path is not None:
                script = model.to_torchscript()
                # save for use in production environment
                torch.jit.save(script, self.model_save_path)
            else:
                warnings.warn('model_save_path is not specified, abort exporting')


def main(args):
    print(args)
    launcher = MultiLabelClassifyLauncher(args)
    launcher.launch(args.stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', required=True, choices=['fit', 'validate', 'test', 'predict', 'export_model'])
    parser.add_argument('-cm', '--compile_model', action='store_true')
    parser.add_argument('-msp', '--model_save_path', default=None, help='TorchScript导出路径')
    parser.add_argument('-vsd', '--viz_save_dir', default=None, help='测试时，可视化保存目录')
    args = parser.parse_args()
    main(args)
