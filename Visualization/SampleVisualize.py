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
import shutil
import glob

from VisualizeUtil import *
from PostProcess import *
from MultiLabelClassifier import *

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
        'data_class_path': ColonoscopyMultiLabelDataModule,
        'data_index_file': None,
        'data_root': input_image_root,
        'sample_weight': None,
        'resize_shape': 336,
        'center_crop_shape': 336,
        'brightness_jitter': None,
        'contrast_jitter': None,
        'saturation_jitter': None,
        'batch_size': 1,
        'num_workers': 12,
        'dry_run': None,
        'model_class_path': MultiLabelClassifier_ViT_L_Patch14_336_Class7,
        'input_shape': (3, 336, 336),
        'num_heads': 8,
        'attention_lambda': 0.3,
        'num_classes': 7,
        'thresh': [0.5, 0.5, 0.5, 0.5],
        'lr': None,
        'epochs': None,
        'momentum': None,
        'weight_decay': None,
        'cls_weight': 0.2,
        'outside_acc_thresh': None,
        'nonsense_acc_thresh': None,
        'test_id_map_file_path': None,
        'test_viz_save_dir': None,
        'seed_everything': None,
        'ckpt_path': ckpt_path,
        'model_save_path': None,
        'pred_save_path': None,
        'compile_model': False
    }
    args = argparse.Namespace(**arg_dict)
    launcher = MultiLabelClassifyLauncher(args)
    pred = launcher.launch('predict')
    return pred


def pipeline(args: argparse.Namespace):
    # 按步骤执行管线
    if args.s:
        # 获取视频元信息[，拆帧]
        fps = []
        for i in range(len(args.video_path)):
            fps.append(extract_frames(args.video_path[i], args.frame_path[i], 1) if args.x else get_video_fps(args.video_path[i]))

        # 实例化模型预测
        pred = []
        for i in range(len(args.video_path)):
            pred.append(predict(args.device, args.frame_path[i], args.ckpt_path))

        # 原始模型预测日志
        for i in range(len(args.video_path)):
            model_predict = None
            if args.pred_json_path is not None and i < len(args.pred_json_path):
                model_predict = log_model_predict(pred[i], args.pred_json_path[i])
            if args.pred_signal_path is not None and i < len(args.pred_signal_path):
                if model_predict is None:
                    model_predict = log_model_predict(pred[i], None)
                plot_model_predict(model_predict, args.pred_signal_path[i])

        # 后处理
        post_label = []
        for i in range(len(args.video_path)):
            signal_mat = parse_predict_label(pred[i])
            scaled_N = [max(1, math.ceil(fps[i] / 25.0 * N)) for N in args.kernel_sizes]
            median_filtered_mat = median_filter_signal_mat(signal_mat, scaled_N)
            post_label.append(legalize_label(median_filtered_mat, signal_mat))

        # 后处理结果日志
        for i in range(len(args.video_path)):
            post_lb = None
            if args.post_json_path is not None and i < len(args.post_json_path):
                post_lb = log_post_predict(post_label[i], args.post_json_path[i])
            if args.post_signal_path is not None and i < len(args.post_signal_path):
                if post_lb is None:
                    post_lb = log_post_predict(post_label[i], None)
                plot_post_label(post_lb, args.post_signal_path[i])

        # 执行渲染例程
        if args.r:
            for i in range(len(args.video_path)):
                draw_frames(args.frame_path[i], args.render_path[i], post_label)
                merge_frames_to_video(args.render_path[i], args.output_path[i], fps)

    # 按样本执行管线
    else:
        for i in range(len(args.video_path)):
            # 获取视频元信息[，拆帧]
            fps = extract_frames(args.video_path[i], args.frame_path[i], 1) if args.x else get_video_fps(args.video_path[i])

            # 实例化模型预测
            pred = predict(args.device, args.frame_path[i], args.ckpt_path)

            # pred: List[Tuple[torch.Tensor, torch.Tensor]] = \
            #     [
            #         (torch.from_numpy(np.array(([[0.1, 0.2, 0.9, 0.1, 0.1, 0.6, 0.1]]))),
            #          torch.from_numpy(np.array(([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])))),
            #         (torch.from_numpy(np.array(([[0.9, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1]]))),
            #          torch.from_numpy(np.array(([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))))
            #     ]

            # 原始模型预测日志
            model_predict = None
            if args.pred_json_path is not None and i < len(args.pred_json_path):
                model_predict = log_model_predict(pred, args.pred_json_path[i])
            if args.pred_signal_path is not None and i < len(args.pred_signal_path):
                if model_predict is None:
                    model_predict = log_model_predict(pred, None)
                plot_model_predict(model_predict, args.pred_signal_path[i])

            # 后处理
            signal_mat = parse_predict_label(pred)
            scaled_N = [max(1, math.ceil(fps / 25.0 * N)) for N in args.kernel_sizes]
            median_filtered_mat = median_filter_signal_mat(signal_mat, scaled_N)
            post_label = legalize_label(median_filtered_mat, signal_mat)

            # 后处理结果日志
            post_lb = None
            if args.post_json_path is not None and i < len(args.post_json_path):
                post_lb = log_post_predict(post_label, args.post_json_path[i])
            if args.post_signal_path is not None and i < len(args.post_signal_path):
                if post_lb is None:
                    post_lb = log_post_predict(post_label, None)
                plot_post_label(post_lb, args.post_signal_path[i])

            # 执行渲染例程
            if args.r:
                draw_frames(args.frame_path[i], args.render_path[i], post_label)
                merge_frames_to_video(args.render_path[i], args.output_path[i], fps)


def main(args: argparse.Namespace):
    raw_dict: Dict = vars(args).copy()

    if args.b:
        video_path_l: List[str] = []
        for e in args.video_ext:
            video_path_l += glob.glob(osp.join(args.video_path[0], '**', f'*.{e}'), recursive=True)
        video_path_l.sort()
        raw_dict['video_path'] = video_path_l
        video_path_rel_l = [osp.relpath(p, args.video_path[0]) for p in video_path_l]
        raw_dict['frame_path'] = [osp.join(args.frame_path[0], osp.splitext(p)[0]) for p in video_path_rel_l]
        if args.pred_json_path is not None:
            raw_dict['pred_json_path'] = [osp.join(args.pred_json_path[0], f'{osp.splitext(osp.basename(p))[0]}_model_pred.json') for p in
                                          video_path_rel_l]
        if args.pred_signal_path is not None:
            raw_dict['pred_signal_path'] = [osp.join(args.pred_signal_path[0], f'{osp.splitext(osp.basename(p))[0]}_model_pred.png') for p in
                                            video_path_rel_l]
        if args.post_json_path is not None:
            raw_dict['post_json_path'] = [osp.join(args.post_json_path[0], f'{osp.splitext(osp.basename(p))[0]}_post_label.json') for p in
                                          video_path_rel_l]
        if args.post_signal_path is not None:
            raw_dict['post_signal_path'] = [osp.join(args.post_signal_path[0], f'{osp.splitext(osp.basename(p))[0]}_post_label.png') for p in
                                            video_path_rel_l]
        raw_dict['render_path'] = [osp.join(args.render_path[0], osp.splitext(p)[0]) for p in video_path_rel_l]
        raw_dict['output_path'] = [osp.join(args.output_path[0], f'{osp.splitext(p)[0]}_with_label{osp.splitext(p)[1]}') for p in video_path_rel_l]

    wrapped_args = argparse.Namespace(**raw_dict)

    # 移除已存在的中转目录
    for fp, rp in zip(wrapped_args.frame_path, wrapped_args.render_path):
        if args.x:
            shutil.rmtree(fp, ignore_errors=True)
        if args.r:
            shutil.rmtree(rp, ignore_errors=True)
    # 单独存储日志的目录不会被移除

    print(f'wrapped_args: {wrapped_args}')
    pipeline(wrapped_args)


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
    parser.add_argument('-b', action='store_true', help='执行批处理')
    parser.add_argument('-x', action='store_true', help='执行拆帧例程')
    parser.add_argument('-r', action='store_true', help='执行渲染例程')
    parser.add_argument('-s', action='store_true', help='步骤执行，按步骤执行管线，而不是按样本执行')
    parser.add_argument('--video_ext', type=str, nargs='+', default=['mp4'], help='视频扩展名过滤器，用于批处理，可输入多个')
    parser.add_argument('--video_path', type=str, nargs='+', help='视频路径，批处理时为视频目录路径')
    parser.add_argument('--frame_path', type=str, nargs='+', help='拆帧目录，批处理时为拆帧上级目录路径')
    parser.add_argument('--pred_json_path', type=str, nargs='+', help='模型预测日志标签Json存储路径，批处理时为存储目录路径', default=None)
    parser.add_argument('--pred_signal_path', type=str, nargs='+', help='模型预测日志信号图存储路径，批处理时为存储目录路径', default=None)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', help='卷积核规格列表（25FPS基准）')
    parser.add_argument('--post_json_path', type=str, nargs='+', help='后处理日志标签Json存储路径，批处理时为存储目录路径', default=None)
    parser.add_argument('--post_signal_path', type=str, nargs='+', help='后处理日志信号图存储路径，批处理时为存储目录路径', default=None)
    parser.add_argument('--render_path', type=str, nargs='+', help='渲染目录，批处理时为渲染上级目录路径')
    parser.add_argument('--output_path', type=str, nargs='+', help='视频输出路径，批处理时为输出目录路径')
    parser.add_argument('--device', type=int, nargs='+', help='设备号')
    parser.add_argument('--ckpt_path', type=str, help='lightning-pytorch模型文件路径')

    args = parser.parse_args()
    print(f'raw_args: {args}')
    main(args)

    # Model: R105_train_vitp14s336c7_400
    # Device: (0) RTX 3090
    # Dataset: /mnt/data/cwy/Datasets/RemedyBubbles
    # nohup python ./Visualization/SampleVisualize.py -bxrs --video_path /mnt/data/cwy/Datasets/RemedyBubbles --frame_path /mnt/data/cwy/RemedyBubbles_Viz/extract --pred_json_path /mnt/data/cwy/RemedyBubbles_Viz/pred_json --pred_signal_path /mnt/data/cwy/RemedyBubbles_Viz/pred_signal --kernel_sizes 121 51 51 51 --post_json_path /mnt/data/cwy/RemedyBubbles_Viz/post_json --post_signal_path /mnt/data/cwy/RemedyBubbles_Viz/post_signal --render_path /mnt/data/cwy/RemedyBubbles_Viz/render --output_path /mnt/data/cwy/RemedyBubbles_Viz/video --device 0 --ckpt_path Experiment/R105_train_vitp14s336c7_400/tensorboard_fit/checkpoints/MuLModel_best_cls4Acc_epoch=039_label_cleansing_acc_thresh=0.9628.ckpt > log/RemedyBubbles_Viz.log &

    # Model: R105_train_vitp14s336c7_400
    # Device: (0) RTX 3090
    # Dataset: /mnt/data/cwy/Datasets/TestClips
    # nohup python ./Visualization/SampleVisualize.py -bxrs --video_path /mnt/data/cwy/Datasets/TestClips --frame_path /mnt/data/cwy/TestClips_Viz/extract --pred_json_path /mnt/data/cwy/TestClips_Viz/pred_json --pred_signal_path /mnt/data/cwy/TestClips_Viz/pred_signal --kernel_sizes 121 51 51 51 --post_json_path /mnt/data/cwy/TestClips_Viz/post_json --post_signal_path /mnt/data/cwy/TestClips_Viz/post_signal --render_path /mnt/data/cwy/TestClips_Viz/render --output_path /mnt/data/cwy/TestClips_Viz/video --device 0 --ckpt_path Experiment/R105_train_vitp14s336c7_400/tensorboard_fit/checkpoints/MuLModel_best_cls4Acc_epoch=039_label_cleansing_acc_thresh=0.9628.ckpt > log/RemedyBubbles_Viz.log &
