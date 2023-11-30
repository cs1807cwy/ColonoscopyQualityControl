import os
import os.path as osp
import json
import cv2
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import MultiLabelClassifier.Modelv3
from MultiLabelClassifier import *
import matplotlib.pyplot as plt


def main():
    for i in range(22):
        logit = np.zeros((10000, 7))
        label = np.zeros((10000, 7))
        X = np.arange(0, logit.shape[0])
        label_name = ['outside', 'nonsense', 'ileocecal', 'bbps0', 'bbps1', 'bbps2', 'bbps3']
        color = ['blue', 'orange', 'red', '#5b0f00', '#7f6000', 'cyan', 'green']
        fig, axs = plt.subplots(len(label_name), 2, figsize=(40, 32), constrained_layout=True)
        fig.suptitle(f'model predicts')
        for i in range(len(label_name)):
            Y = logit[:, i]
            axs[i][0].plot(X, Y, color=color[i])
            axs[i][0].set_title(f'{label_name[i]} signal')
            axs[i][0].set_ylim(-0.2, 1.2)
            Y = label[:, i]
            axs[i][1].plot(X, Y, color=color[i])
            axs[i][1].set_title(f'{label_name[i]} label')
            axs[i][1].set_ylim(-0.2, 1.2)
        plt.close(fig)


if __name__ == '__main__':
    main()
