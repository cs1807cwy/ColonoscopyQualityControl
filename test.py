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
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

print(os.environ)

def main():
    fig, axs = plt.subplots(2, 2, figsize=(40, 32), constrained_layout=True)


if __name__ == '__main__':
    main()
