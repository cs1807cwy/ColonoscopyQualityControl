"""
    可视化功能函数
    可视化相关的辅助函数都在此实现：
        1.视频拆帧
        2.标签渲染到帧
"""
import glob

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Union, Tuple
import os
import os.path as osp
import cv2

""" 
    视频拆帧
"""

def extract_frames(input_video_root: str, input_video_ext: list, frame_save_root: str, step: int) -> Dict[str, float]:
    # 筛选全部具有ext指定包含后缀名的文件
    items = []
    video_info = dict()
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
        video_info[name] = fps
        while success:
            success, frame = cap.read()
            if not success:
                break
            if frame_count % step == 0:
                cv2.imwrite(osp.join(frame_save_path, f"{frame_count:06d}.png"), frame)
            frame_count += 1
        cap.release()
        print(f"Video {name} : Split into {frame_count} frames.")
    return video_info


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


def draw_label_color_block_on_frame(imageSrcPath: str, imageSavePath: str, **labels) -> Image:
    frame: Image = Image.open(imageSrcPath)
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
    os.makedirs(osp.dirname(imageSavePath), exist_ok=True)
    frame.save(imageSavePath)
    return frame


# 测试代码
if __name__ == '__main__':
    imageSrcPath = 'log/image.png'
    imageSavePath = 'log/out.png'
    draw_label_color_block_on_frame(
        imageSrcPath,
        imageSavePath,
        **{'outside': True,
           'nonsense': True,
           'ileocecal': True,
           'bbps': 3})
    viz = cv2.imread(imageSavePath)
    cv2.imshow('viz', viz)
    cv2.waitKey()
    cv2.destroyAllWindows()
