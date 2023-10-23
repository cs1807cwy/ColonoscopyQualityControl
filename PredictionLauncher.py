import warnings
import os.path as osp

import importlib
import torch
import lightning

import argparse
from lightning.pytorch import Trainer
from Classifier import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

# hparams for Trainer
accelerator = 'cuda'
device = 1

# hparams for DataModule
data_root = '/mnt/data/cwy/Datasets/UIHNJMuL'
ext = ['png', 'jpg']

resize_shape: Tuple[int, int] = (336, 336)
center_crop_shape: Tuple[int, int] = (336, 336)

# global settings
seed_everything = 0


class SimpleClassifyLauncher:
    def __init__(self, args):
        # hparams for Trainer
        self.accelerator = args.accelerator
        self.device = args.device

        # hparams for DataModule
        self.data_class_path = args.data_class_path
        self.data_root = args.data_root
        self.ext = args.data_ext
        self.resize_shape = args.resize_shape
        self.center_crop_shape = args.center_crop_shape

        # hparams for Module
        self.model_class_path = args.model_class_path

        # global settings
        self.seed_everything = args.seed_everything
        self.ckpt_path = args.ckpt_path

        if self.seed_everything is not None:
            lightning.seed_everything(self.seed_everything)

        # custom settings
        self.compile_model = args.compile_model
        self.pred_save_path = args.pred_save_path

    def get_trainer(self) -> Trainer:
        trainer = Trainer(accelerator=self.accelerator, devices=[self.device])
        return trainer

    def get_data(self) -> LightningDataModule:
        data = self.data_class_path(
            dataset_root=self.data_root,
            ext=self.ext,
            resize_shape=self.resize_shape,
            center_crop_shape=self.center_crop_shape,
            batch_size=1,
            num_workers=12,
        )
        return data

    def get_model(self) -> LightningModule:
        model = self.model_class_path(batch_size=1)
        return model

    def launch(self):
        model = self.get_model()
        target_device = f'{self.accelerator}:{self.device}'
        if self.accelerator == 'cpu':
            target_device = 'cpu'
        model = model.load_from_checkpoint(
                self.ckpt_path,
                map_location=torch.device(target_device),
                batch_size=1,
        )
        if self.compile_model:
            model = torch.compile(model, mode='default')  # mode=['default', 'reduce-overhead', 'max-autotune']
        data = self.get_data()
        trainer = self.get_trainer()

        pred = trainer.predict(
                model=model,
                datamodule=data,
                return_predictions=True)
        items = []
        for e in ext:
            items += glob.glob(osp.join(self.data_root, '**', f'*.{e}'), recursive=True)

        items = sorted(items)
        assert len(items) == len(pred)


        if self.pred_save_path is not None:
            os.makedirs(osp.dirname(self.pred_save_path), exist_ok=True)
            pred_json = {}  # {img_path(str): pred(int)}
            items = [item.replace(self.data_root, '').strip('/') for item in items]  # remove root path

            for item, p in zip(items, pred):
                pred_json[item] = int(p)
            with open(self.pred_save_path, 'w') as fp:
                json.dump(pred_json, fp, indent=2)
        else:
            for item, p in zip(items, pred):
                print(f'{item}: {p}')
            warnings.warn('pred_save_path is None, so the prediction results are printed on the screen.')

def get_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_class_one_exp(module_class_name: str):
    head, ext = osp.splitext(module_class_name)
    if ext == '':
        return globals()[head]
    else:
        return get_class(head, ext[1:])

def print_args(parser: Union[str, argparse.ArgumentParser], args: argparse.Namespace, only_non_default: bool = False):
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
        if not only_non_default:
            print(default_str)

    print('-' * 60)


def main(parser: argparse.ArgumentParser, args: argparse.Namespace):
    print_args(parser, args, only_non_default=False)

    arg_dict: Dict = vars(args)
    arg_dict.update({
        'data_class_path': get_class_one_exp(args.data_class_path),
        'model_class_path': get_class_one_exp(args.model_class_path)
    })

    args = argparse.Namespace(**arg_dict)
    print_args('Resolved Args', args)

    launcher = SimpleClassifyLauncher(args)
    launcher.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="肠镜多任务质控启动器",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 自定义参数
    parser.add_argument('-cm', '--compile_model', action='store_true',
                        help='编译模型以加速(使用GPU，要求CUDA Compute Capability >= 7.0)')
    parser.add_argument('-psp', '--pred_save_path', default=None, help='预测结果保存路径，置空时不保存')

    # 全局控制参数
    parser.add_argument('-se', '--seed_everything', type=int, default=seed_everything, help='随机种子')
    parser.add_argument('-cp', '--ckpt_path', default=None, help='模型检查点路径')

    # 训练器参数
    parser.add_argument('-acc', '--accelerator', default=accelerator, choices=['cpu', 'gpu', 'tpu', 'ipu', 'auto'],
                        help='加速器')
    parser.add_argument('-dev', '--device', type=int, default=device, help='设备号')

    # 数据装载器参数
    parser.add_argument('-dcp', '--data_class_path', help='数据模型类路径')
    parser.add_argument('-de', '--data_ext', nargs='+', default=ext, help='数据集图片扩展名')
    parser.add_argument('-dr', '--data_root', default=data_root, help='数据集根路径')
    parser.add_argument('-rs', '--resize_shape', type=int, nargs=2, default=resize_shape,
                        help='预处理时缩放图像目标规格；格式：(H, W)')
    parser.add_argument('-ccs', '--center_crop_shape', type=int, nargs=2, default=center_crop_shape,
                        help='中心裁剪规格，配合resize_shape使用可裁去边缘；格式：(H, W)（注：只有中心(H, W)区域进入网络，以匹配主干网络的输入规格）')

    # 网络模型参数
    parser.add_argument('-mcp', '--model_class_path', help='网络模型类路径')

    main(parser, parser.parse_args())
