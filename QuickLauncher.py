import warnings
import os.path as osp

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
experiment_name = 'R001_Releasev1_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch1000'
ckpt_every_n_epochs = 20

# hparams for DataModule
data_class_path = ColonoscopyMultiLabelDataModule
data_root = '/mnt/data/cwy/Datasets/UIHNJMuL'
# Loc split Ref
data_index_file = '/mnt/data/cwy/Datasets/UIHNJMuL/folds/fold0.json'
sample_weight = {
    'ileocecal': 4800,
    'nofeature': 4800,
    'nonsense': 480,
    'outside': 96,
}
# cleansing split Ref
# data_index_file = '/mnt/data/cwy/Datasets/UIHNJMuL/cls_folds/fold0.json'
# sample_weight = {
#     'nobbps': 1800,
#     'bbps0': 900,
#     'bbps1': 900,
#     'bbps2': 900,
#     'bbps3': 900,
# }
resize_shape = (224, 224)
center_crop_shape = (224, 224)
brightness_jitter = 0.8
contrast_jitter = 0.8
saturation_jitter = 0.8
batch_size = 48
num_workers = 16
dry_run = False

# hparams for Module
model_class_path = MultiLabelClassifier_ViT_L_Patch16_224_Class7
input_shape = (224, 224)  # 主干网络固定输入规格为(224, 224)，请勿修改！
num_heads = 8
attention_lambda = 0.3
num_classes = 7
thresh = 0.5
lr = 0.0001
epochs = max_epochs
momentum = 0.9
weight_decay = 0.0001
cls_weight = 1.0
outside_acc_thresh = 0.9
nonsense_acc_thresh = 0.9

# global settings
seed_everything = 0


class MultiLabelClassifyLauncher:
    def __init__(self, args):
        # hparams for Trainer
        self.accelerator = args.accelerator
        self.strategy = args.strategy
        self.devices = args.devices
        self.max_epochs = args.max_epochs
        self.check_val_every_n_epoch = args.check_val_every_n_epoch
        self.log_every_n_steps = args.log_every_n_steps
        self.default_root_dir = args.default_root_dir
        self.logger = args.logger
        self.callbacks = args.callbacks

        # hparams for DataModule
        self.data_class_path = args.data_class_path
        self.data_index_file = args.data_index_file
        self.data_root = args.data_root
        self.sample_weight = args.sample_weight
        self.resize_shape = args.resize_shape
        self.center_crop_shape = args.center_crop_shape
        self.brightness_jitter = args.brightness_jitter
        self.contrast_jitter = args.contrast_jitter
        self.saturation_jitter = args.saturation_jitter
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dry_run = args.dry_run

        # hparams for Module
        self.model_class_path = args.model_class_path
        self.input_shape = args.input_shape
        self.num_heads = args.num_heads
        self.attention_lambda = args.attention_lambda
        self.num_classes = args.num_classes
        self.thresh = args.thresh
        # batch_size
        self.lr = args.lr
        self.epochs = args.max_epochs
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.cls_weight = args.cls_weight
        self.outside_acc_thresh = args.outside_acc_thresh
        self.nonsense_acc_thresh = args.nonsense_acc_thresh
        self.viz_save_dir = args.viz_save_dir

        # global settings
        self.seed_everything = args.seed_everything
        self.ckpt_path = args.ckpt_path

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
            outside_acc_thresh=self.outside_acc_thresh,
            nonsense_acc_thresh=self.nonsense_acc_thresh,
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


def print_args(parser: Union[str, argparse.ArgumentParser], args: argparse.Namespace, only_non_defaut: bool = False):
    args_dict = vars(args)
    if isinstance(parser, str):
        arg_list = ['=' * 20 + f' {parser} ' + '=' * 20]
        for k, v in args_dict.items():
            arg_list.append('{}: {}'.format(k, v))
        args_str = '\n'.join(arg_list)
        print(args_str)
    elif isinstance(parser, argparse.ArgumentParser):
        default_str_list = ['=' * 20 + ' Default Args ' + '=' * 20]
        non_default_str_list = ['=' * 20 + ' Not Default Args ' + '=' * 20]

        for k, v in args_dict.items():
            default = parser.get_default(k)
            if v == default:
                default_str_list.append('{}: {}'.format(k, v))
            else:
                non_default_str_list.append('{}: {} (default: {})'.format(k, v, default))

        default_str = '\n'.join(default_str_list)
        non_default_str = '\n'.join(non_default_str_list)

        print(non_default_str)
        if not only_non_defaut:
            print(default_str)

    print('-' * 60)


def main(parser: argparse.ArgumentParser, args: argparse.Namespace):
    print_args(parser, args, only_non_defaut=False)

    arg_dict: Dict = vars(args)
    arg_dict.update({
        'default_root_dir': osp.join('Experiment', args.experiment_name),
        # region logger
        'logger': [
            TensorBoardLogger(
                save_dir='Experiment',
                name=args.experiment_name,
                version='v1'
            ),
            CSVLogger(
                save_dir='Experiment',
                name=args.experiment_name,
                version='v1'
            )
        ],
        # endregion
        # region callbacks
        'callbacks': [
            TQDMProgressBar(refresh_rate=args.tqdm_refresh_rate),
            ModelCheckpoint(
                save_last=True,
                monitor='epoch',
                mode='max',
                every_n_epochs=args.ckpt_every_n_epochs,  # 每n个epochs保存一个检查点
                filename='WMuL_{epoch:03d}',
                save_top_k=args.max_epochs // args.ckpt_every_n_epochs
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
                monitor='label_ileocecal_acc_thresh',
                mode='max',
                filename='WMuL_best_ileoAcc_{epoch:03d}_{label_ileocecal_acc_thresh:.4f}'
            ),
            ModelCheckpoint(
                monitor='label_ileocecal_prec_thresh',
                mode='max',
                filename='WMuL_best_ileoPrec_{epoch:03d}_{label_ileocecal_prec_thresh:.4f}'
            ),
            ModelCheckpoint(
                monitor='label_cleansing_acc_thresh',
                mode='max',
                filename='WMuL_best_cls4Acc_{epoch:03d}_{label_cleansing_acc_thresh:.4f}'
            ),
            ModelCheckpoint(
                monitor='label_cleansing_biclassify_acc_thresh',
                mode='max',
                filename='WMuL_best_cls2Acc_{epoch:03d}_{label_cleansing_biclassify_acc_thresh:.4f}'
            ),
        ],
        # endregion
        'data_class_path': data_class_path,
        'sample_weight': {k: v for k, v in zip(args.sample_weight_key, args.sample_weight_value)},
        'center_crop_shape': center_crop_shape,
        'dry_run': dry_run,
        'model_class_path': model_class_path,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'epochs': args.max_epochs,
    })

    args = argparse.Namespace(**arg_dict)
    print_args('Resolved Args', args)

    launcher = MultiLabelClassifyLauncher(args)
    if args.stage != 'arg_debug':
        launcher.launch(args.stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 自定义参数
    parser.add_argument('-s', '--stage', required=True, choices=['fit', 'validate', 'test', 'predict', 'export_model', 'arg_debug'],
                        help='运行模式：fit-训练(包含训练时验证)，validate-验证，test-测试，predict-预测，export_model-导出TorchScript模型，arg_debug-仅检查参数')
    parser.add_argument('-cm', '--compile_model', action='store_true', help='编译模型以加速(使用GPU，要求CUDA Compute Capability >= 7.0)')
    parser.add_argument('-msp', '--model_save_path', default=None, help='TorchScript导出路径，置空时不导出')

    # 全局控制参数
    parser.add_argument('-se', '--seed_everything', type=int, default=seed_everything, help='随机种子')
    parser.add_argument('-me', '--max_epochs', type=int, default=max_epochs, help='训练纪元总数')
    parser.add_argument('-bs', '--batch_size', type=int, default=batch_size, help='批大小')
    parser.add_argument('-cp', '--ckpt_path', default=None, help='预训练模型路径，置空时不装载')

    # 训练器参数
    parser.add_argument('-acc', '--accelerator', default=accelerator, choices=['cpu', 'gpu', 'tpu', 'ipu', 'auto'], help='加速器')
    parser.add_argument('-str', '--strategy', default=strategy, choices=['ddp', 'ddp_spawn', 'ddp_notebook'], help='运行策略')
    parser.add_argument('-dev', '--devices', type=int, nargs='+', default=devices, help='设备号')
    parser.add_argument('-cve', '--check_val_every_n_epoch', type=int, default=check_val_every_n_epoch,
                        help='验证纪元间隔，1表示每个训练纪元运行一次验证流程')
    parser.add_argument('-ls', '--log_every_n_steps', type=int, default=log_every_n_steps, help='日志记录间隔，1表示每个迭代轮次记录一次日志')
    parser.add_argument('-en', '--experiment_name', default=experiment_name, help='实验名称，用于生成实验目录')
    parser.add_argument('-ce', '--ckpt_every_n_epochs', type=int, default=ckpt_every_n_epochs,
                        help='检查点保存间隔，1表示每个训练纪元保存一次检查点')
    parser.add_argument('-trr', '--tqdm_refresh_rate', type=int, default=20, help='进度条刷新间隔，1表示每个迭代轮次进行一次刷新')

    # 数据装载器参数
    parser.add_argument('-dif', '--data_index_file', default=data_index_file, help='数据集索引文件')
    parser.add_argument('-dr', '--data_root', default=data_root, help='数据集根路径')
    parser.add_argument('-swk', '--sample_weight_key', nargs='+', default=list(sample_weight.keys()), help='重采样数据子集列表')
    parser.add_argument('-swv', '--sample_weight_value', type=int, nargs='+', default=list(sample_weight.values()),
                        help='重采样数量列表(与sample_weight_key一一对应)')
    parser.add_argument('-rs', '--resize_shape', type=int, nargs=2, default=resize_shape,
                        help='预处理时缩放图像目标规格，可用于裁去边缘（注：只有中心(224, 224)区域进入网络，以匹配主干网络的输入规格）')
    parser.add_argument('-bj', '--brightness_jitter', type=float, default=brightness_jitter,
                        help='标准化亮度泛化域宽，[max(0, 1 - brightness), 1 + brightness]')
    parser.add_argument('-cj', '--contrast_jitter', type=float, default=contrast_jitter,
                        help='标准化对比度泛化域宽，[max(0, 1 - contrast), 1 + contrast]')
    parser.add_argument('-sj', '--saturation_jitter', type=float, default=saturation_jitter,
                        help='标准化饱和度泛化域宽，[max(0, 1 - saturation), 1 + saturation]')
    parser.add_argument('-nw', '--num_workers', type=int, default=num_workers, help='数据装载线程数')

    # 网络模型参数
    parser.add_argument('-nh', '--num_heads', type=int, default=num_heads, choices=[1, 2, 4, 6, 8], help='输出头（不同温度T）数量')
    parser.add_argument('-al', '--attention_lambda', type=float, default=attention_lambda, help='输出头类特征权重')
    parser.add_argument('-thr', '--thresh', type=float, default=thresh, help='逐类标签置信度阈值')
    parser.add_argument('-lr', '--lr', type=float, default=lr, help='SGD优化器学习率')
    parser.add_argument('-mom', '--momentum', type=float, default=momentum, help='SGD优化器动量')
    parser.add_argument('-wd', '--weight_decay', type=float, default=weight_decay, help='SGD优化器权重衰退')
    parser.add_argument('-cw', '--cls_weight', type=float, default=cls_weight, help='清洁度损失权重')
    parser.add_argument('-oat', '--outside_acc_thresh', type=float, default=outside_acc_thresh, help='outside性能筛选线')
    parser.add_argument('-nat', '--nonsense_acc_thresh', type=float, default=nonsense_acc_thresh, help='nonsense性能筛选线')
    parser.add_argument('-vsd', '--viz_save_dir', default=None, help='测试时，分类错误图像的保存目录，置空时不保存')

    main(parser, parser.parse_args())
    # nohup python MultiLabelTrain.py -s fit -cm > log/R001_Releasev1_train_MultiLabelClassifier_ViT_L_patch16_224_compile_epoch1000.log &
