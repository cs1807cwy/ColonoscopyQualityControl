import torch
import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from MultiLabelClassifier import *

# hparams for Trainer
accelerator = 'gpu'
strategy = 'ddp'
devices = [2, 3]
max_epochs = 500
log_every_n_steps = 10
default_root_dir = 'Experiment/502_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch500'
logger = [
    TensorBoardLogger(
        save_dir='Experiment',
        name='502_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch500',
        version='tensorboard_train_val'
    ),
    CSVLogger(
        save_dir='Experiment',
        name='502_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch500',
        version='csv_train_val'
    )
]
callbacks = [
    TQDMProgressBar(refresh_rate=20),
    ModelCheckpoint(
        save_last=True,
        monitor='epoch',
        mode='max',
        every_n_epochs=1,
        filename='CleansingClassifier_{epoch:03d}',
        save_top_k=1
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
image_index_file_or_root = '/mnt/data4/cwy/Datasets/UIHWMuL/folds/fold0.json'
sample_weight = {
    'ileocecal': 4800,
    'nofeature': 4800,
    'nonsense': 4800,
    'outside': 4800,
}
resize_shape = (224, 224)
center_crop_shape = (224, 224)
brightness_jitter = 0.8
contrast_jitter = 0.8
saturation_jitter = 0.8
batch_size = 32
num_workers = 8
dry_run = False

# hparams for Module
model_class_path = MultiLabelClassifier_ViT_L_Patch16_224_Class7
input_shape = (224, 224)
num_heads = 8
attention_lambda = 0.3
num_classes = 7
thresh = 0.5
# batch_size
lr = 0.0001
momentum = 0.9
weight_decay = 0.0001
epochs = max_epochs


def get_trainer() -> Trainer:
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=default_root_dir
    )
    return trainer


def get_data() -> LightningDataModule:
    data = data_class_path(
        data_index_file=image_index_file_or_root,
        sample_weight=sample_weight,
        resize_shape=resize_shape,
        center_crop_shape=center_crop_shape,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        saturation_jitter=saturation_jitter,
        batch_size=batch_size,
        num_workers=num_workers,
        dry_run=dry_run
    )
    return data


def get_model() -> LightningModule:
    model = model_class_path(
        input_shape=input_shape,
        num_heads=num_heads,
        attention_lambda=attention_lambda,
        num_classes=num_classes,
        thresh=thresh,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        epochs=epochs
    )
    return model


def main(args):
    model = torch.compile(get_model(), mode='max-autotune')
    data = get_data()
    trainer = get_trainer()

    if args.stage == 'fit':
        trainer.fit(
            model=model,
            datamodule=data,
            ckpt_path=args.ckpt_path
        )
    elif args.stage == 'validate':
        trainer.validate(
            model=model,
            datamodule=data,
            ckpt_path=args.ckpt_path
        )
    elif args.stage == 'test':
        trainer.test(
            model=model,
            datamodule=data,
            ckpt_path=args.ckpt_path
        )
    elif args.stage == 'predict':
        trainer.predict(
            model=model,
            datamodule=data,
            ckpt_path=args.ckpt_path
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', required=True, choices=['fit', 'validate', 'test', 'predict'])
    parser.add_argument('-c', '--ckpt_path', default=None)
    args = parser.parse_args()
    main(args)
