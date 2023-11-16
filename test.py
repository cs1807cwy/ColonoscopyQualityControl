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

if __name__ == '__main__':
    print(osp.abspath(os.curdir))
    weights_dict = torch.load(
        r'F:\CBIBF3\storage\PycharmProjects\ColonoscopyQualityControl\Experiment\Archive\R105_train_vitp14s336c7_400\tensorboard_fit\checkpoints\MuLModel_best_cls4Acc_epoch=039_label_cleansing_acc_thresh=0.9628.ckpt',
        map_location='cuda:0')
    print(weights_dict.keys())
    print(weights_dict['hyper_parameters'])
    weights_dict['hyper_parameters']['thresh'] = [0.1, 0.2, 0.3, 0.5]
    print(weights_dict)
    torch.save(weights_dict, r'F:\CBIBF3\storage\MuLModel_best_cls4Acc_threshmod.ckpt')
    model = MultiLabelClassifier_ViT_L_Patch14_336_Class7()
    model = model.load_from_checkpoint(r'F:\CBIBF3\storage\MuLModel_best_cls4Acc_threshmod.ckpt', map_location='cuda:0')
    print(model.hparams)
