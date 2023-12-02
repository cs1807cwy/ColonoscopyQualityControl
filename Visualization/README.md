# 可视化管线

实现基于PyTorch、Lightning框架、Matplotlib、Numpy、Python-Opencv、Pillow

[TOC]

## 目录结构

```bash
Visualization
├─SampleVisualize.py  # 可视化管线启动入口
├─VisualizeUtil.py  # 可视化功能函数
└─README.md  # 当前说明文档
```

## 安装依赖

安装绘图、图像处理相关组件：

```bash
conda install matplotlib numpy python-opencv pillow
```

## 启动管线

程序入口位于：

```bash
Visualization/SampleVisualize.py
```

启动命令示例：

```bash
python Visualization/SampleVisualize.py
-bxrms # 以批处理模式启动，执行拆帧、渲染帧、视频合成例程，按步骤执行管线
--video_path Datasets/TestClips
--frame_path TestClips_Viz/extract
--pred_json_path TestClips_Viz/pred_json
--pred_signal_path TestClips_Viz/pred_signal
# 依次表示[体内/外，好/坏帧，是/非回盲部，清洁度]标签的滤波核大小(像素窗口宽度)
--kernel_sizes 121 51 51 51
--post_json_path TestClips_Viz/post_json
--post_signal_path TestClips_Viz/post_signal
--render_path TestClips_Viz/render
--output_path TestClips_Viz/video
--device 0
--ckpt_path Experiment/R105_train_vitp14s336c7_400/tensorboard_fit/checkpoints/MuLModel_best_cls4Acc_epoch=039_label_cleansing_acc_thresh=0.9628.ckpt
```

启动命令参数详细用法：

```bash
usage: SampleVisualize.py [-h] [-b] [-x] [-r] [-m] [-s]
                          [--video_ext VIDEO_EXT [VIDEO_EXT ...]]
                          [--video_path VIDEO_PATH [VIDEO_PATH ...]]
                          [--frame_path FRAME_PATH [FRAME_PATH ...]]
                          [--pred_json_path PRED_JSON_PATH [PRED_JSON_PATH ...]]
                          [--pred_signal_path PRED_SIGNAL_PATH [PRED_SIGNAL_PATH ...]]
                          --kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]
                          [--post_json_path POST_JSON_PATH [POST_JSON_PATH ...]]
                          [--post_signal_path POST_SIGNAL_PATH [POST_SIGNAL_PATH ...]]
                          [--render_path RENDER_PATH [RENDER_PATH ...]]
                          [--output_path OUTPUT_PATH [OUTPUT_PATH ...]]
                          --device DEVICE [DEVICE ...] --ckpt_path CKPT_PATH

启动可视化管线，请确保各路径参数列表的长度一致

options:
  -h, --help            show this help message and exit
  -b, --batching        执行批处理 (default: False)
  -x, --extract_frame   执行拆帧例程 (default: False)
  -r, --render_frame    执行渲染帧例程 (default: False)
  -m, --merge_video     执行视频合成例程 (default: False)
  -s, --step_mode       步骤执行，按步骤执行管线，而不是按样本执行 (default: False)
  --video_ext VIDEO_EXT [VIDEO_EXT ...]
                        视频扩展名过滤器，用于批处理，可输入多个 (default: ['mp4'])
  --video_path VIDEO_PATH [VIDEO_PATH ...]
                        视频路径，批处理时为视频目录路径 (default: None)
  --frame_path FRAME_PATH [FRAME_PATH ...]
                        拆帧目录，批处理时为拆帧上级目录路径 (default: None)
  --pred_json_path PRED_JSON_PATH [PRED_JSON_PATH ...]
                        模型预测日志标签Json存储路径，批处理时为存储目录路径 (default: None)
  --pred_signal_path PRED_SIGNAL_PATH [PRED_SIGNAL_PATH ...]
                        模型预测日志信号图存储路径，批处理时为存储目录路径 (default: None)
  --kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]
                        卷积核规格列表（25FPS基准） (default: None)
  --post_json_path POST_JSON_PATH [POST_JSON_PATH ...]
                        后处理日志标签Json存储路径，批处理时为存储目录路径 (default: None)
  --post_signal_path POST_SIGNAL_PATH [POST_SIGNAL_PATH ...]
                        后处理日志信号图存储路径，批处理时为存储目录路径 (default: None)
  --render_path RENDER_PATH [RENDER_PATH ...]
                        渲染目录，批处理时为渲染上级目录路径 (default: None)
  --output_path OUTPUT_PATH [OUTPUT_PATH ...]
                        视频输出路径，批处理时为输出目录路径 (default: None)
  --device DEVICE [DEVICE ...]
                        设备号 (default: None)
  --ckpt_path CKPT_PATH
                        lightning-pytorch模型文件路径 (default: None)
```
