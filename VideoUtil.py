import os, glob
import os.path as osp
from typing import Dict

import cv2


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