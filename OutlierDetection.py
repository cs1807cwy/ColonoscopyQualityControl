import argparse
import json
import os
import os.path as osp
import cv2
import glob
import subprocess
import numpy as np
import math
from typing import List, Dict


def extract_frames(input_video_root: str, input_video_ext: list, frame_save_root: str) -> Dict[str, float]:
    # 筛选全部具有ext指定包含后缀名的文件
    items = []
    video = dict()
    for e in input_video_ext:
        items += glob.glob(osp.join(input_video_root, '**', f'*.{e}'), recursive=True)
        items = sorted(items)
    for item in items:
        name = item.split('/')[-1].split('.')[0]
        frame_save_path = osp.join(frame_save_root, name)
        os.makedirs(frame_save_path, exist_ok=True)
        cap = cv2.VideoCapture(item)
        frame_count = 0
        success = True
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video {name} : FPS is {fps}.")
        video[name] = fps
        while success:
            success, frame = cap.read()
            if not success:
                break
            cv2.imwrite(osp.join(frame_save_path, f"{frame_count:06d}.png"), frame)
            frame_count += 1
        cap.release()
        print(f"Video {name} : Split into {frame_count} frames.")
    return video


def call_predict_once(exp_name: str, ckpt_path: str, batch_size: int, input_image_root: str,
                      pred_save_path: str) -> int:
    # 调用模型进行预测
    # QuickLauncher.py
    # --stage predict --batch_size 1 --ckpt_path Experiment/R105_train_vitp14s336c7_400/tensorboard_fit/checkpoints/MuLModel_best_ileoPrec_epoch=109_label_ileocecal_prec_thresh=0.9812.ckpt --accelerator gpu --strategy ddp --devices 2 --log_every_n_steps 10 --experiment_name R105_predict_fps_vitp14s336c7_400 --version test_predict --tqdm_refresh_rate 20 --data_class_path MultiLabelClassifier.DataModule.ColonoscopyMultiLabelDataModule --data_root ../Datasets/pred_img --resize_shape 336 336 --center_crop_shape 336 336 --num_workers 12 --model_class_path MultiLabelClassifier.Modelv3.MultiLabelClassifier_ViT_L_Patch14_336_Class7 --pred_save_path Experiment/R105_predict_fps_vitp14s336c7_400/predict_result.json --num_heads 8 --attention_lambda 0.3 --thresh 0.5

    p = subprocess.run([
        '/home/shr/.conda/envs/cwypy310pt20/bin/python', 'QuickLauncher.py',
        '--stage', 'predict',
        '--batch_size', str(batch_size),
        '--ckpt_path', ckpt_path,
        '--accelerator', 'gpu',
        '--strategy', 'ddp',
        '--devices', '2',
        '--log_every_n_steps', '10',
        '--experiment_name', exp_name,
        '--version', 'detect',
        '--tqdm_refresh_rate', '20',
        '--data_class_path', 'MultiLabelClassifier.DataModule.ColonoscopyMultiLabelDataModule',
        '--data_root', input_image_root,
        '--resize_shape', '336', '336',
        '--center_crop_shape', '336', '336',
        '--num_workers', '12',
        '--model_class_path', 'MultiLabelClassifier.Modelv3.MultiLabelClassifier_ViT_L_Patch14_336_Class7',
        '--pred_save_path', pred_save_path,
        '--num_heads', '8',
        '--attention_lambda', '0.3',
        '--thresh', '0.5'
    ])
    return p.returncode


def call_predict_all(exp_name: str, ckpt_path: str, batch_size: int, frame_save_root: str, pred_save_root: str):
    for v in sorted((os.listdir(frame_save_root))):
        pred_save_path = osp.join(pred_save_root, v, 'predict_result.json')
        os.makedirs(osp.dirname(pred_save_path), exist_ok=True)
        r_code = call_predict_once(exp_name, ckpt_path, batch_size, osp.join(frame_save_root, v), pred_save_path)
        print(f"Video {v} : Predict return {r_code}.")


def detect_outlier_all(pred_save_root: str, video_fps_info: Dict[str, float], outlier_thresh_scale: float) -> \
        Dict[str, Dict[int, Dict[int, List[List[int]]]]]:
    # 读取预测结果，进行异常检测
    # predict_result.json 格式：
    # {"000000.png": [0, 0, 0, 0, 0, 0, 1], "000001.png": [0, 0, 0, 0, 0, 0, 1],...}
    # 每个子列表为连续的每一帧的预测结果，其中每个子列表的第i个元素为第i个标签的预测结果
    # 要求每个标签的预测结果(0或者1)连续相同的帧数大于等于阈值threshold，否则认为这个连续范围内的帧为异常帧
    # 返回值为异常帧的起始和终止帧的索引（整个列表中的位置索引），格式为[[start1, end1], [start2, end2], ...]
    total_outlier = dict()
    for v in sorted(os.listdir(pred_save_root)):
        frame_threshold = round(video_fps_info[v] * outlier_thresh_scale)
        print(f"Video {v} : Frame threshold is {frame_threshold}.")
        pred_save_path = osp.join(pred_save_root, v, 'predict_result.json')
        with open(pred_save_path, 'r') as f:
            pred_result = json.load(f)
        pred_array = np.array([pred_result[k] for k in sorted(pred_result.keys())])
        pred_array = np.transpose(pred_array)
        # 此时pred_array的shape为(7, n)，其中n为视频帧数
        # 接下来对每一行进行异常检测，判断连续的相同0片段或者连续的相同1片段的帧数是否大于等于frame_threshold
        # 如果不大于，则认为这个连续范围内的帧为异常帧
        # 返回值为异常帧的起始和终止帧的索引Dict,key为v, value为list, 其shape为(7)，表示7个类别, 每个元素为一个列表，格式为[[start1, end1], [start2, end2], ...]
        curr_video_outlier = dict()
        for i in range(pred_array.shape[0]):
            # 求连续相同的0片段和1片段的范围，并判断是否大于等于阈值
            curr_label_outlier = {0: [], 1: []}
            curr_label = pred_array[i, 0]  # 0 or 1
            curr_start = 0
            curr_end = 0
            for j in range(pred_array.shape[1]):
                if pred_array[i, j] == curr_label:
                    curr_end = j
                else:
                    if curr_end - curr_start + 1 <= frame_threshold:
                        if curr_label == 0:
                            curr_label_outlier[0].append([curr_start, curr_end])
                        else:
                            curr_label_outlier[1].append([curr_start, curr_end])
                    curr_label = pred_array[i, j]
                    curr_start = j
                    curr_end = j
            curr_video_outlier[i] = curr_label_outlier
        total_outlier[v] = curr_video_outlier
    return total_outlier


def save_outlier(outlier_result: Dict[str, List[List[int]]], outlier_save_root: str):
    os.makedirs(outlier_save_root, exist_ok=True)
    json.dump(outlier_result, open(osp.join(outlier_save_root, 'outlier_result.json'), 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--experiment_name', help='实验名称，用于生成实验目录')
    parser.add_argument('-ivr', '--input_video_root', type=str, help='输入视频路径')
    parser.add_argument('-ext', '--input_video_ext', type=str, nargs='+', default=['mp4'], help='输入视频后缀名')
    parser.add_argument('-fsr', '--frame_save_root', type=str, help='视频抽帧保存路径')
    parser.add_argument('-psr', '--pred_save_root', type=str, help='预测结果保存路径')
    parser.add_argument('-osr', '--outlier_save_root', type=str, help='异常帧保存路径')
    parser.add_argument('-cp', '--ckpt_path', type=str, help='模型权重路径')
    parser.add_argument('-dev', '--devices', type=str, nargs='+', default=[4,7,8,9], help='使用的GPU设备')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('-ots', '--outlier_thresh_scale', type=float, default=0.11, help='异常帧阈值缩放倍率')

    args = parser.parse_args()
    video_fps_info = extract_frames(args.input_video_root, args.input_video_ext, args.frame_save_root)
    call_predict_all(args.experiment_name, args.ckpt_path, args.batch_size, args.frame_save_root, args.pred_save_root)
    total_outlier = detect_outlier_all(args.pred_save_root, video_fps_info, args.outlier_thresh_scale)
    save_outlier(total_outlier, args.outlier_save_root)
