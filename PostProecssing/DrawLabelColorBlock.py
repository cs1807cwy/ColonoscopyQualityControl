import PIL
from PIL import Image, ImageDraw, ImageFont
import os
import os.path as osp
import cv2


def DrawLabelColorBlockOnFrame(imageSrcPath: str, imageSavePath: str, **kwarg) -> Image:
    frame: Image = Image.open(imageSrcPath)
    height = frame.height
    width = frame.width
    block_height: int = min(height // 8, width)
    draw: ImageDraw = ImageDraw.ImageDraw(frame)

    # specified font size
    font = ImageFont.truetype(r'Font/等线Bold.ttf', block_height)

    # 绘制 outside 标签
    x, y = (0, 0)
    is_outside: bool = kwarg['outside']
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill='blue' if is_outside else 'black')
    draw.text((x, y), '外' if is_outside else '', align='center', font=font)

    # 绘制 nonsense 标签
    y += block_height
    is_nonsense: bool = kwarg['nonsense']
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill='orange' if is_nonsense else 'black')
    draw.text((x, y), '坏' if is_nonsense else '', align='center', font=font)

    # 绘制 ileocecal 标签
    y += block_height
    is_ileocecal: bool = kwarg['ileocecal']
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill='red' if is_ileocecal else 'black')
    draw.text((x, y), '盲' if is_ileocecal else '', align='center', font=font)

    # 绘制 bbps 标签
    y += block_height
    cls_bbps: int = kwarg['bbps']
    draw.rectangle(((x, y), (x + block_height, y + block_height)), fill=['#5b0f00', '#7f6000', 'cyan', 'green', 'black'][cls_bbps])
    draw.text((x + block_height // 4, y), f'{cls_bbps}' if cls_bbps >= 0 else '', align='center', font=font)

    frame.save(imageSavePath)
    return frame


if __name__ == '__main__':
    imageSrcPath = 'log/image.png'
    imageSavePath = 'log/out.png'
    DrawLabelColorBlockOnFrame(
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
