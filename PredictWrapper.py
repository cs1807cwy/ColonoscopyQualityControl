import os
import os.path as osp
import shutil
import subprocess
import sys


def call_predict_once(exp_name: str, devices: str, ckpt_path: str, input_image_root: str,
                      pred_save_path: str) -> int:
    # 调用模型进行预测
    # QuickLauncher.py
    # --stage predict --batch_size 1 --ckpt_path Experiment/R105_train_vitp14s336c7_400/tensorboard_fit/checkpoints/MuLModel_best_ileoPrec_epoch=109_label_ileocecal_prec_thresh=0.9812.ckpt --accelerator gpu --strategy ddp --devices 2 --log_every_n_steps 10 --experiment_name R105_predict_fps_vitp14s336c7_400 --version test_predict --tqdm_refresh_rate 20 --data_class_path MultiLabelClassifier.DataModule.ColonoscopyMultiLabelDataModule --data_root ../Datasets/pred_img --resize_shape 336 336 --center_crop_shape 336 336 --num_workers 12 --model_class_path MultiLabelClassifier.Modelv3.MultiLabelClassifier_ViT_L_Patch14_336_Class7 --pred_save_path Experiment/R105_predict_fps_vitp14s336c7_400/predict_result.json --num_heads 8 --attention_lambda 0.3 --thresh 0.5
    py_exec = sys.executable
    p = subprocess.run([
        py_exec, 'QuickLauncher.py',
        '--stage', 'predict',
        '--batch_size', '1',
        '--ckpt_path', ckpt_path,
        '--accelerator', 'gpu',
        '--strategy', 'ddp',
        '--devices', str(devices),
        '--log_every_n_steps', '10',
        '--experiment_name', exp_name,
        '--version', 'detect',
        '--tqdm_refresh_rate', '20',
        '--data_class_path', 'MultiLabelClassifier.DataModule.ColonoscopyMultiLabelDataModule',
        '--data_root', input_image_root,
        '--resize_shape', '336', '336',
        '--center_crop_shape', '336', '336',
        '--num_workers', '12',
        '--model_class_path', 'MultiLabelClassifier.Modelv3.MultiLabelClassifier_ViT_L_Patch14_336_Class7',
        '--pred_save_path', pred_save_path,
        '--num_heads', '8',
        '--attention_lambda', '0.3',
        '--thresh', '0.5'
    ])
    return p.returncode


def call_predict_all(exp_name: str, devices: str, ckpt_path: str, frame_save_root: str,
                     pred_save_root: str):
    for v in sorted(os.listdir(frame_save_root)):
        pred_save_path = osp.join(pred_save_root, v, 'predict_result.json')
        if osp.exists(pred_save_path):
            os.remove(pred_save_path)
        os.makedirs(osp.dirname(pred_save_path), exist_ok=True)
        print(f"Video {v} : Start predicting.")
        r_code = call_predict_once(exp_name, devices, ckpt_path, osp.join(frame_save_root, v), pred_save_path)
        if r_code == 0:
            shutil.copy2(pred_save_path, osp.join(pred_save_root, v, f'predict_result_{osp.basename(ckpt_path)}.json'))
        print(f"Video {v} : Predict return {r_code}.")
