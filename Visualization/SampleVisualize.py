"""
    可视化程序入口
    实现可视化管线：（对管线中的每一步封装一个单独函数）
        1.读取视频,拆帧 --> 指定源视频路径，提取元信息，指定存储全部帧的目录
        2.实例化模型进行预测 --> （只读，可选参数）指定预测标签Json存储路径（logit、label都记录在一个Json里），指定信号图存储路径（logit 7个、label 7个标签信号，14合1的总图）
        3.解析预测结果 --> 变换成后处理所需的数据格式
        4.后处理 --> （只读，可选参数）调用后处理方法生成结果，输出到Json文件（指定标签Json存储路径，只记录label），指定信号图存储路径（label 4个标签信号，4合1的总图）
        5.帧渲染 --> 指定存储渲染帧的目录，调用帧渲染方法生成结果，
        6.合成视频 --> 指定合成视频的目录，视频规格与源视频的元信息保持一致

    注：
        管线中输出的任何Json仅作日志使用，请勿做数据读取之用
        管线中的下一步处理直接从上一步的输出中获取数据，无需通过Json中转
"""
import argparse
import math

from VisualizeUtil import *
from PostProcess import *

from QuickLauncher import MultiLabelClassifyLauncher

"""
    3.模型预测日志标签Json格式：
    {
        'logit': [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3],
            ...
        ],
        'label': [
            [0, 0, 1, 0, 0, 0, 1] # [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3],
            ...
        ]
    }
"""

"""
    5.后处理日志标签Json格式：
    [
        [0, 0, 0, 3] # [outside, nonsense, ileocecal, bbps(-1 for nobbps)],
        ...
    ]
"""


def predict(devices: list, input_image_root: str, ckpt_path: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    arg_dict = {
        'accelerator': 'gpu',
        'strategy': 'ddp',
        'devices': devices,
        'max_epochs': None,
        'check_val_every_n_epoch': None,
        'log_every_n_steps': 10,
        'default_root_dir': None,
        'logger': None,
        'callbacks': None,
        'data_class_path': 'MultiLabelClassifier.DataModule.ColonoscopyMultiLabelDataModule',
        'data_index_file': None,
        'data_root': input_image_root,
        'sample_weight': None,
        'resize_shape': '336',
        'center_crop_shape': '336',
        'brightness_jitter': None,
        'contrast_jitter': None,
        'saturation_jitter': None,
        'batch_size': 1,
        'num_workers': 12,
        'dry_run': None,
        'model_class_path': 'MultiLabelClassifier.Modelv3.MultiLabelClassifier_ViT_L_Patch14_336_Class7',
        'input_shape': None,
        'num_heads': 8,
        'attention_lambda': 0.3,
        'num_classes': None,
        'thresh': ['0.5', '0.5', '0.5', '0.5'],
        'lr': None,
        'epochs': None,
        'momentum': None,
        'weight_decay': None,
        'cls_weight': None,
        'outside_acc_thresh': None,
        'nonsense_acc_thresh': None,
        'test_id_map_file_path': None,
        'test_viz_save_dir': None,
        'seed_everything': None,
        'ckpt_path': ckpt_path,
        'model_save_path': None,
        'pred_save_path': None,
        'compile_model': None
    }
    args = argparse.Namespace(**arg_dict)
    launcher = MultiLabelClassifyLauncher(args)
    pred = launcher.launch("predict")
    return pred


def pipeline(args: argparse.Namespace):
    for i in range(len(args.video_path)):
        fps = extract_frames(args.video_path[i], args.frame_path[i], 1)
        pred = predict(args.device, args.frame_path[i], args.ckpt_path)
        signal_mat = parse_predict_label(pred)
        scaled_N = [max(1, math.ceil(fps / 25.0 * N)) for N in args.kernel_sizes]
        median_filtered_mat = median_filter_signal_mat(signal_mat, scaled_N)
        post_label = legalize_label(median_filtered_mat, signal_mat)
        draw_frames(args.frame_path[i], args.render_path[i], post_label)
        merge_frames_to_video(args.render_path[i], args.output_path[i], fps)


if __name__ == '__main__':
    """
        实例化 ArgumentParser，接收下列参数：[*]表示可选参数
            源视频路径 List[str]：指定每个视频路径
            拆帧目录：List[str]：每个视频指定一个拆帧目录
            [模型预测日志标签Json存储路径]：List[str]：每个视频指定一个Json存储路径
            [模型预测日志信号图存储路径]：List[str]：每个视频指定一个信号图存储路径
            卷积核规格列表 List[int*4]：分别对应 [体外，坏帧，回盲部，清洁度] 的中值滤波1D卷积核规格
            [后处理日志标签Json存储路径] List[str]：每个视频指定一个Json存储路径
            [后处理日志信号图存储路径] List[str]：每个视频指定一个信号图存储路径
            渲染目录：List[str]：每个视频指定一个渲染帧目录
            视频输出路径 List[str]：每个视频指定一个输出路径
            设备号 List[int]：指定每个视频的设备号
            模型预测检查点路径 str：指定模型预测检查点路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, nargs='+', help='视频路径')
    parser.add_argument('--frame_path', type=str, nargs='+', help='拆帧目录')
    parser.add_argument('--pred_json_path', type=str, nargs='+', help='模型预测日志标签Json存储路径', default=None)
    parser.add_argument('--pred_signal_path', type=str, nargs='+', help='模型预测日志信号图存储路径', default=None)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', help='卷积核规格列表')
    parser.add_argument('--post_json_path', type=str, nargs='+', help='后处理日志标签Json存储路径', default=None)
    parser.add_argument('--post_signal_path', type=str, nargs='+', help='后处理日志信号图存储路径', default=None)
    parser.add_argument('--render_path', type=str, nargs='+', help='渲染目录')
    parser.add_argument('--output_path', type=str, nargs='+', help='视频输出路径')
    parser.add_argument('--device', type=int, nargs='+', help='设备号')
    parser.add_argument('--ckpt_path', type=str, help='模型预测检查点路径')
