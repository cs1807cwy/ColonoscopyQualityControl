import argparse
import json
import math

from VideoUtil import *
from PredictWrapper import *
import numpy as np
from typing import List, Dict


def detect_outlier_all(pred_save_root: str, std25_frame_thresholds: dict, video_info: dict) -> \
        Dict[str, Dict[int, Dict[int, List[List[int]]]]]:
    # 读取预测结果，进行异常检测
    # predict_result.json 格式：
    # {"000000.png": [0, 0, 0, 0, 0, 0, 1], "000001.png": [0, 0, 0, 0, 0, 0, 1],...}
    # 每个子列表为连续的每一帧的预测结果，其中每个子列表的第i个元素为第i个标签的预测结果
    # 要求每个标签的预测结果(0或者1)连续相同的帧数大于等于阈值threshold，否则认为这个连续范围内的帧为异常帧
    # 返回值为异常帧的起始和终止帧的索引（整个列表中的位置索引），格式为[[start1, end1], [start2, end2], ...]
    total_outlier = dict()
    if osp.isfile(pred_save_root):
        search_list = [pred_save_root]
    else:
        search_list = sorted(os.listdir(pred_save_root))
        search_list = [osp.join(pred_save_root, v, 'predict_result.json') for v in search_list]

    for v in search_list:
        with open(v, 'r') as f:
            pred_result = json.load(f)
        video_name = osp.basename(osp.dirname(v))

        # 根据实际帧率计算帧阈值, video_info的格式为{video_name: fps}, std25_frame_thresholds表示是25fps时的帧阈值
        video_fps = video_info[video_name]
        frame_thresholds = dict()
        for k in std25_frame_thresholds:
            frame_thresholds[k] = math.ceil(std25_frame_thresholds[k] * video_fps / 25.0)

        print(f"Video {video_name} : Frame threshold -> {frame_thresholds}")
        pred_array_np = np.array([pred_result[k] for k in sorted(pred_result.keys())])
        pred_array_ori = np.transpose(pred_array_np)

        # pred_array[1] += pred_array[0]  # 将outside标签视为nonsense标签处理

        # 此时pred_array的shape为(7, n)，其中n为视频帧数
        # 接下来对每一行进行异常检测，判断连续的相同0片段或者连续的相同1片段的帧数是否大于等于frame_threshold
        # 如果不大于，则认为这个连续范围内的帧为异常帧
        # 返回值为异常帧的起始和终止帧的索引Dict,key为v, value为list, 其shape为(7)，表示7个类别, 每个元素为一个列表，格式为[[start1, end1], [start2, end2], ...]
        curr_video_outlier = dict()
        for i in range(pred_array_ori.shape[0]):
            # 求连续相同的0片段和1片段的范围，并判断是否大于等于阈值
            curr_frame_threshold = frame_thresholds[i]
            curr_label_outlier = {0: [], 1: []}
            pred_array = np.copy(pred_array_ori)
            if i == 0:
                pred_array[1] += pred_array[0]  # 将outside标签视为nonsense标签处理

            curr_label = pred_array[i, 0]  # 0 or 1
            curr_start = 0
            curr_end = 0
            for j in range(pred_array.shape[1]):
                if pred_array[i, j] == curr_label:
                    curr_end = j
                else:
                    if curr_end - curr_start + 1 <= curr_frame_threshold:
                        if curr_label == 0:
                            curr_label_outlier[0].append([curr_start, curr_end])
                        else:
                            curr_label_outlier[1].append([curr_start, curr_end])
                    curr_label = pred_array[i, j]
                    curr_start = j
                    curr_end = j
            curr_video_outlier[i] = curr_label_outlier
        total_outlier[video_name] = curr_video_outlier
    return total_outlier


def save_outlier_all(outlier_result: Dict[str, Dict[int, Dict[int, List[List[int]]]]], outlier_save_root: str):
    os.makedirs(outlier_save_root, exist_ok=True)
    # json.dump(outlier_result, open(osp.join(outlier_save_root, 'outlier_result.json'), 'w'), indent=2)
    for video in outlier_result:
        video_outlier = outlier_result[video]
        video_outlier_path = osp.join(outlier_save_root, video)
        os.makedirs(video_outlier_path, exist_ok=True)
        json.dump(video_outlier, open(osp.join(video_outlier_path, 'outlier_result.json'), 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--experiment_name', help='实验名称，用于生成实验目录')
    parser.add_argument('-ivr', '--input_video_root', type=str, help='输入视频路径')
    parser.add_argument('-ext', '--input_video_ext', type=str, nargs='+', default=['mp4'], help='输入视频后缀名')
    parser.add_argument('-fsr', '--frame_save_root', type=str, help='视频抽帧保存路径')
    parser.add_argument('-psr', '--pred_save_root', type=str, help='预测结果保存路径')
    parser.add_argument('-osr', '--outlier_save_root', type=str, help='异常帧保存路径')
    parser.add_argument('-cp', '--ckpt_path', type=str, help='模型权重路径')
    parser.add_argument('-dev', '--devices', type=str, default=0, help='使用的GPU设备')
    # parser.add_argument('-ots', '--outlier_thresh_scale', type=float, default=0.11, help='异常帧阈值缩放倍率')

    args = parser.parse_args()
    video_info = extract_frames(args.input_video_root, args.input_video_ext, args.frame_save_root, 1)
    print(video_info)
    call_predict_all(args.experiment_name, args.devices, args.ckpt_path, args.frame_save_root, args.pred_save_root, video_info)
    std25_frame_threshes = {0: 100, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6}
    total_outlier = detect_outlier_all(args.pred_save_root, std25_frame_threshes, video_info)
    save_outlier_all(total_outlier, args.outlier_save_root)
