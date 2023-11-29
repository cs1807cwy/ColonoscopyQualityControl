import os
import os.path as osp
import json
import cv2
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import MultiLabelClassifier.Modelv3
from MultiLabelClassifier import *


def main():
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA is not available. Please make sure CUDA drivers are installed.")


if __name__ == '__main__':
    main()
