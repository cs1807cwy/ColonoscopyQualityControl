import torchvision.transforms as transforms
import numpy as np
from PIL import Image

if __name__ == '__main__':
    sample = np.arange(0, 256).reshape((16, 16))
    print(sample)

    img = Image.fromarray(sample.astype('uint8'))
    # print(np.array(img))

    ts = transforms.ToTensor()(img)
    print(ts)

    # new = bright * (old + contrast * (mean - old))

    ss = transforms.ColorJitter(brightness=[0.5, 0.5], contrast=[0.5, 0.5])(ts)
    print(np.array(ss))

    # ss = transforms.ColorJitter(brightness=[1,1])(ts)
    # print(np.array(ss))
    # ss = transforms.ColorJitter(brightness=[2,2])(ts)
    # print(np.array(ss))
    # ss = transforms.ColorJitter(brightness=[0.5,0.5])(ts)
    # print(np.array(ss))
    # ss = transforms.ColorJitter(contrast=[1,1])(ts)
    # print(np.array(ss))
    # ss = transforms.ColorJitter(contrast=[2,2])(ts)
    # print(np.array(ss))
    # ss = transforms.ColorJitter(contrast=[0.5,0.5])(ts)
    # print(np.array(ss))

    # img = Image.open(r'E:\Datasets\UIHIMG\ileocecal\ileocecal_000007.png')
    # img = Image.open(r'E:\Datasets\UIHIMG\nofeature\nofeature_006510.png')
    img = Image.open(r'E:\Datasets\UIHIMG\ileocecal\ileocecal_000699.png')
    # img = Image.open(r'E:\Datasets\UIHIMG\outside\outside_000057.png')
    # img.show()

    ts = (img)
    # print(ts)

    transFunc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((306, 306)),
        transforms.CenterCrop(256),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation([90, 90])])
    ])

    for i in range(10):
        shape = ss.shape
        ss = transFunc(ts)
        img = ss.cpu().clone()
        img = img.squeeze(0)  # 压缩一维
        img = transforms.ToPILImage()(img)  # 自动转换为0-255
        # img.show()
        img.save(f'{i}.png')
