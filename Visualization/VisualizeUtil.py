"""
    可视化功能函数
    可视化相关的辅助函数都在此实现：
        1.视频拆帧
        2.标签渲染到帧
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Union, Tuple
import os
import os.path as osp
import cv2

""" 
    视频拆帧
"""

def extract_frames(input_video_path: str, frame_save_root: str, step: int=1) -> float:
    # 筛选全部具有ext指定包含后缀名的文件
    item = input_video_path
    name = item.split('/')[-1].split('.')[0]
    frame_save_path = osp.join(frame_save_root, name)
    os.makedirs(frame_save_path, exist_ok=True)
    cap = cv2.VideoCapture(item)
    frame_count = 0
    success = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video {name} : FPS is {fps}.")
    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % step == 0:
            cv2.imwrite(osp.join(frame_save_path, f"{frame_count:06d}.png"), frame)
        frame_count += 1
    cap.release()
    print(f"Video {name} : Split into {frame_count} frames.")
    return fps


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
    print(frame_names)
    os.makedirs(render_frame_dir, exist_ok=True)
    render_count: int = min(len(frame_names), post_label.shape[0])
    if len(frame_names) != post_label.shape[0]:
        print('not equal, use min')
    for i in range(render_count):
        abs_src_path: str = osp.join(raw_frame_dir, frame_names[i])
        abs_out_path: str = osp.join(render_frame_dir, frame_names[i])
        label_seq: List[int] = list(post_label[i])
        labels: Dict[str, int] = {
            'outside': label_seq[0],
            'nonsense': label_seq[1],
            'ileocecal': label_seq[2],
            'bbps': label_seq[3]
        }
        draw_label_color_block_on_frame(abs_src_path, abs_out_path, **labels)


def merge_frames_to_video(render_frame_dir: str, output_path: str, fps: float):
    frame_names: List[str] = os.listdir(render_frame_dir)
    frame_names.sort(key=lambda x: int(x.split('.')[0]))

    head: np.ndarray = cv2.imread(osp.join(render_frame_dir, frame_names[0]))
    height, width, layers = head.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame_i in frame_names:
        frame: np.ndarray = cv2.imread(osp.join(render_frame_dir, frame_i))
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    post_label = np.array([
        [0, 1, 0, -1],
        [0, 0, 0, 3]
    ])
    draw_frames('/mnt/data/cwy/ColonoscopyQualityControl/Experiment/R106_predict_vitp14s336c7/frames/ZJY_10fps',
                '/mnt/data/cwy/ColonoscopyQualityControl/Experiment/R106_predict_vitp14s336c7/frames/extest', post_label)
