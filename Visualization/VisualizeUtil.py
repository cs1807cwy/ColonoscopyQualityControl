"""
    可视化功能函数
    可视化相关的辅助函数都在此实现：
        1.视频拆帧
        2.标签渲染到帧
"""
import warnings
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Union, Tuple
import os
import os.path as osp
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from PostProcess import extract_predict_logit_label_7

""" 
    视频拆帧
"""


def extract_frames(input_video_path: str, frame_save_root: str, step: int = 1) -> float:
    item = input_video_path
    name = item.split('/')[-1].split('.')[0]
    os.makedirs(frame_save_root, exist_ok=True)
    cap = cv2.VideoCapture(item)
    frame_count = 0
    success = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Video {name} : FPS is {fps}. '
          'Start frame extraction...')
    with tqdm(total=frames, desc='extract frame') as pbar:
        while success:
            success, frame = cap.read()
            if not success:
                break
            if frame_count % step == 0:
                cv2.imwrite(osp.join(frame_save_root, f"{frame_count:06d}.png"), frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"Video {name} : Split into {frame_count} frames.")
    return fps


def get_video_fps(input_video_path: str) -> float:
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video {osp.basename(osp.splitext(input_video_path)[0])} : FPS is {fps}.')
    return fps


"""
    输出模型原始预测日志
    包含 logit 和 label
    模型预测日志标签Json格式：
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


def log_model_predict(predict_label: List[Tuple[torch.Tensor, torch.Tensor]], log_path: str) -> Dict[str, List[List[Union[float, int]]]]:
    logit_list, label_list = extract_predict_logit_label_7(predict_label)
    pred_dict: Dict[str, List[List[Union[float, int]]]] = {
        'logit': logit_list,
        'label': [[int(e) for e in lb] for lb in label_list]
    }
    if (log_path is not None):
        log_path = osp.abspath(log_path)
        os.makedirs(osp.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as fp:
            json.dump(pred_dict, fp, indent=2)
        print(f'model predict json log: {log_path}')
    return pred_dict


"""
    输出模型原始预测信号图
    包含 logit 和 label
    输入数据格式：
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


def plot_model_predict(data: Dict[str, List[List[Union[float, int]]]], plot_path: str):
    if plot_path is None:
        return

    logit: np.ndarray = np.array(data['logit'])  # Mat [frame_count, 7]
    label: np.ndarray = np.array(data['label'])  # Mat [frame_count, 7]

    plt.rcParams['font.size'] = 32
    plt.figure()
    X = np.arange(0, logit.shape[0])
    label_name = ['outside', 'nonsense', 'ileocecal', 'bbps0', 'bbps1', 'bbps2', 'bbps3']
    color = ['blue', 'orange', 'red', '#5b0f00', '#7f6000', 'cyan', 'green']

    fig, axs = plt.subplots(len(label_name), 2, figsize=(40, 32), constrained_layout=True)
    fig.suptitle(f'{osp.splitext(osp.basename(plot_path))[0]} model predicts')
    for i in range(len(label_name)):
        Y = logit[:, i]
        axs[i][0].plot(X, Y, color=color[i])
        axs[i][0].set_title(f'{label_name[i]} signal')
        axs[i][0].set_ylim(-0.2, 1.2)
        Y = label[:, i]
        axs[i][1].plot(X, Y, color=color[i])
        axs[i][1].set_title(f'{label_name[i]} label')
        axs[i][1].set_ylim(-0.2, 1.2)

    plot_path = osp.abspath(plot_path)
    os.makedirs(osp.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f'plot model predict signal: {plot_path}')


"""
    输出后处理结果日志
    包含 label
    后处理日志标签Json格式：
    [
        [0, 0, 0, 3] # [outside, nonsense, ileocecal, bbps(-1 for nobbps)],
        ...
    ]
"""


def log_post_predict(label_mat: np.ndarray, log_path: str) -> List[List[int]]:
    pred_dict: List[List[int]] = label_mat.tolist()
    if log_path is not None:
        log_path = osp.abspath(log_path)
        os.makedirs(osp.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as fp:
            json.dump(pred_dict, fp, indent=2)
        print(f'post label json log: {log_path}')
    return pred_dict


"""
    输出后处理结果信号图
    包含 label
    输入数据格式：
    [
        [0, 0, 0, 3] # [outside, nonsense, ileocecal, bbps(-1 for nobbps)],
        ...
    ]
"""


def plot_post_label(data: List[List[int]], plot_path: str):
    if plot_path is None:
        return

    label: np.ndarray = np.array(data)  # Mat [frame_count, 7]

    plt.rcParams['font.size'] = 32
    plt.figure()
    X = np.arange(0, label.shape[0])
    label_name = ['outside', 'nonsense', 'ileocecal', 'bbps']
    color = ['blue', 'orange', 'red', '#7f6000']
    y_bottom = [-0.2, -1.2, -1.2, -1.2]
    y_top = [1.2, 1.2, 1.2, 3.2]

    fig, axs = plt.subplots(len(label_name), 1, figsize=(40, 32), constrained_layout=True)
    fig.suptitle(f'{osp.splitext(osp.basename(plot_path))[0]} post signals')
    for i in range(len(label_name)):
        Y = label[:, i]
        axs[i].plot(X, Y, color=color[i])
        axs[i].set_title(f'{label_name[i]} label')
        axs[i].set_ylim(y_bottom[i], y_top[i])

    plot_path = osp.abspath(plot_path)
    os.makedirs(osp.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f'plot post label signal: {plot_path}')


""" 
    将标签渲染到帧
    参数 labels: 形如
    {
        'outside': 0,
        'nonsense': 0,
        'ileocecal': 0,
        'bbps': 3
    }
"""


def draw_label_color_block_on_frame(image_src_path: str, image_save_path: str, **labels) -> Image:
    frame: Image = Image.open(image_src_path)
    height = frame.height
    width = frame.width
    block_height: int = min(height // 8, width)
    draw: ImageDraw = ImageDraw.ImageDraw(frame)

    # specified font size
    font = ImageFont.truetype(r'Font/等线Bold.ttf', block_height)

    # 绘制 outside 标签
    x, y = (0, 0)
    is_outside: bool = labels['outside'] == 1
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill='blue' if is_outside else 'black')
    draw.text((x, y), '外' if is_outside else '', align='center', font=font)

    # 绘制 nonsense 标签
    y += block_height
    is_nonsense: bool = labels['nonsense'] == 1
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill='orange' if is_nonsense else 'black')
    draw.text((x, y), '坏' if is_nonsense else '', align='center', font=font)

    # 绘制 ileocecal 标签
    y += block_height
    is_ileocecal: bool = labels['ileocecal'] == 1
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill='red' if is_ileocecal else 'black')
    draw.text((x, y), '盲' if is_ileocecal else '', align='center', font=font)

    # 绘制 bbps 标签
    y += block_height
    cls_bbps: int = labels['bbps']
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill=['#5b0f00', '#7f6000', 'cyan', 'green', 'black'][cls_bbps])
    draw.text((x + block_height // 4, y), f'{cls_bbps}' if cls_bbps >= 0 else '', align='center', font=font)
    os.makedirs(osp.dirname(image_save_path), exist_ok=True)
    frame.save(image_save_path)
    return frame


# post_label Mat [frame_count, 4]
def draw_frames(raw_frame_dir: str, render_frame_dir: str, post_label: np.ndarray):
    frame_names: List[str] = os.listdir(raw_frame_dir)
    frame_names.sort()
    os.makedirs(render_frame_dir, exist_ok=True)
    render_count: int = min(len(frame_names), post_label.shape[0])
    if len(frame_names) != post_label.shape[0]:
        warnings.warn('frame count not equal to label count, use min count')
    with tqdm(total=render_count, desc='plot labels on frame') as pbar:
        for i in range(render_count):
            abs_src_path: str = osp.join(raw_frame_dir, frame_names[i])
            abs_out_path: str = osp.join(render_frame_dir, frame_names[i])
            label_seq: List[int] = post_label[i].astype(int).tolist()
            labels: Dict[str, int] = {
                'outside': label_seq[0],
                'nonsense': label_seq[1],
                'ileocecal': label_seq[2],
                'bbps': label_seq[3]
            }
            draw_label_color_block_on_frame(abs_src_path, abs_out_path, **labels)
            pbar.update(1)


def merge_frames_to_video(render_frame_dir: str, output_path: str, fps: float):
    frame_names: List[str] = os.listdir(render_frame_dir)
    frame_names.sort(key=lambda x: int(x.split('.')[0]))

    head: np.ndarray = cv2.imread(osp.join(render_frame_dir, frame_names[0]))
    height, width, layers = head.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    with tqdm(total=len(frame_names), desc='render video') as pbar:
        for frame_i in frame_names:
            frame: np.ndarray = cv2.imread(osp.join(render_frame_dir, frame_i))
            video_writer.write(frame)
            pbar.update(1)
    video_writer.release()


if __name__ == '__main__':
    post_label = np.array([
        [0, 1, 0, -1],
        [0, 0, 0, 3]
    ])
    draw_frames('/mnt/data/cwy/ColonoscopyQualityControl/Experiment/R106_predict_vitp14s336c7/frames/ZJY_10fps',
                '/mnt/data/cwy/ColonoscopyQualityControl/Experiment/R106_predict_vitp14s336c7/frames/extest', post_label)
