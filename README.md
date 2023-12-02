# Colonoscopy Multi-Task Quality Control

# 肠镜多任务质控

It is an implementation based on PyTorch and Lightning

实现基于PyTorch和Lightning框架

[Tutorials](Visualization/README.md) for visualization pipeline 

可视化管线的使用请参见[这里](Visualization/README.md)

[TOC]

## Project Structure

## 工程结构

```bash
<ProjectRoot>
├─.run  # 运行配置文件存储目录，用于命令行参数快速启动器，PyCharm可识别格式
├─Config  # 运行配置文件存储目录，用于Lightning CLI部署启动器（备用）
├─Experiment  # 实验日志、性能、检查点存储目录（按实验名称组织）
│ ├─<ExperimentName 1>  # 实验目录
│ └─<ExperimentName ...>
├─Export  # 导出模型文件存储目录
│ └─<ExportModel>.ths  # 导出的模型文件
├─Font    #渲染用字体文件
├─MultiLabelClassifier  # 神经网络管线目录
│ ├─Network  # 网络模块目录
│ | ├─ClassifyHead.py  # CSRA模块
│ | └─ViT_Backbone.py  # 主干网络（使用Timm迁移模型）
│ ├─DataModule.py  # 数据管理模型
│ ├─Dataset.py  # 数据预处理
│ ├─Modelv2.py  # ViT-L-Patch16-224 CSRA网络模型
│ └─Modelv3.py  # ViT-L-Patch14-336 CSRA网络模型
├─PostProcess   # 后处理例程目录
│ └─PostProcess.py  # 后处理功能函数，5个后处理相关的辅助函数都在此实现
├─Visualzation  # 可视化管线目录
│ ├─SampleVisualize.py  # 可视化管线启动入口
│ └─VisualizeUtil.py  # 可视化功能函数
├─DeployLauncher.py  # Lightning CLI部署启动器（备用启动入口）
├─QuickLauncher.py  # 命令行参数快速启动器（启动入口）
└─README.md  # 当前说明文档
```

## Validated Environment

## 测试环境

Already tested on:

已测试于环境：

```
python 3.10.10
pytorch 2.0.1
pytorch-cuda 11.8
torchvision 0.15.2
lightning 2.0.1.post0
pytorch-lightning 2.0.2
tensorboardx 2.2
timm 0.9.2
Nvidia Driver Version 525.105.17
CUDA Version 12.0 
4 GPU Nvidia GeForce GTX 1080ti
```

```
python 3.10.11
pytorch 2.0.1
pytorch-cuda 11.8
torchvision 0.15.2
lightning 2.0.3
pytorch-lightning 2.0.3
tensorboardx 2.2
timm 0.9.2
Nvidia Driver Version 460.84
CUDA Version 11.2 
2 GPU Nvidia GeForce RTX 3090
```

## Prepare Runtime Environment

## 环境配置

Config new environment using Anaconda

使用Anaconda配置运行环境

Create Virtual Environment `<VirtualEnv>` and install Python3.10 interpreter:

创建虚拟环境`<VirtualEnv>`，安装Python3.10解释器：

```bash
conda create -n <VirtualEnv> python=3.10 -y
```

Activate Environment:

进入环境：

```bash
conda activate <VirtualEnv>
```

Install PyTorch2.0 related components:

安装PyTorch2.0相关组件：（请确认对应CUDA版本，此处以11.8为例）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Install Lightning related components:

安装Lightning相关组件：

```bash
conda install lightning "pytorch-lightning[extra]" -c conda-forge -y
```

Install supporting components (model zoo, logger):

安装模型库、日志记录器等辅助支持：

```bash
conda install tensorboardX timm -c conda-forge -y
```

Install utility packages:

安装实用库：

```bash
conda install numpy pyyaml tqdm pillow -y
```

## Quick Launch

## 快速启动

Entry of the program at:

程序入口位于：

```bash
<ProjectRoot>/QuickLauncher.py
```

Launch command example:

启动命令示例：请重点关注标注为**[核心参数]**的命令选项

**注**：PyTorch2.0启用编译加速功能要求GPU Compute Capability > 7.0，参阅[CUDA GPUs - Compute Capability | NVIDIA Developer](https://developer.nvidia.com/cuda-gpus)

**注**：导出Torch Script可用，export_model_torch_script模式导出可用性已通过验证（可适配演示程序）

```bash
python QuickLauncher.py
# [核心参数] 模式，fit表示以训练模式启动
# 总计7种模式{fit-训练,finetune-优化训练,validate-验证,test-测试,predict-预测,export_model_torch_script-导出TorchScript模型,arg_debug-命令参数调试}
--stage fit
# 使用PyTorch2.0的编译加速（如无需使用，则省略此参数）
--compile_model
# [核心参数] TorchScript导出路径，仅在export_model模式下生效，置空时不导出
--model_save_path Export/some_export.ths
# 设定随机数种子为0，此设定用于保障训练可复现性
--seed_everything 0
# [核心参数] 最大训练轮次，此处设定为400，表示训练400 Epochs后停止训练
--max_epochs 400
# [核心参数] 模型检查点路径，用于恢复和继续训练，置空时不装载
--ckpt_path Experiment/R103_train_vitp14s336c7_400/tensorboard_fit/checkpoints/some_record.ckpt
# [核心参数] 输入数据批大小，此处设定为16，表示每批次传入16个样本（如显存不足，请酌情减小）
--batch_size 16
# [核心参数] 加速设备，gpu表示GPU加速，cpu表示CPU加速
--accelerator gpu
# [非核心参数，保持默认即可] ddp表示使用数据分布并行调度策略
--strategy ddp
# [核心参数] 加速设备指定：accelerator为gpu时，传入所使用的设备编号列表（空格分隔，如此处"2 3"表示使用编号2、3的两个GPU）；accelerator为cpu时，传入加速使用的cpu核心数（如"4"表示使用四核加速）
--devices 2 3
# [非核心参数，保持默认即可] 1表示每1 Epoch后运行一次验证流程来监视模型性能
--check_val_every_n_epoch 1 
# [非核心参数，保持默认即可] 10表示每10批次（Step）训练后更新一次日志
--log_every_n_steps 10
# 实验名称，可自行命名（不带空格），程序将据此创建"Experiment/<实验名称>"目录用于存放数据文件
--experiment_name R103_train_vitp14s336c7_400
# 实验版本号，可自行命名（不带空格），程序将据此创建"Experiment/<实验名称>/<日志版本>"目录用于存放数据文件
--version v1
# 检查点间隔，50表示每经过50 Epochs记录一个模型检查点（检查点用于恢复训练和导出模型）
--ckpt_every_n_epochs 50
# [非核心参数，保持默认即可] 20表示每20批次（Step）训练后更新一次TQDM进度条显示
--tqdm_refresh_rate 20
# [核心参数] 数据模型DataModule类路径（当前仅有一个可用类，请保持默认）
--data_class_path MultiLabelClassifier.DataModule.ColonoscopyMultiLabelDataModule
# [核心参数] 数据集索引文件路径，仅支持UIHNJMuLv3发布包索引文件格式，参见以下路径UIHNJMuLv3/cls_folds/fold0.json
--data_index_file ../Datasets/UIHNJMuLv3/cls_folds/fold0.json
# [核心参数] 数据集根路径，必须是data_index_file索引文件所使用的根路径（此处是UIHNJMuLv3发布包根目录）
--data_root ../Datasets/UIHNJMuLv3
# [核心参数] 数据重采样标签，必须枚举data_index_file索引文件中的标签，此处是一个无清洁度标签nobbps和四级BBPS标签
--sample_weight_key nobbps bbps0 bbps1 bbps2 bbps3
# [核心参数] 数据重采样率，与sample_weight_key数量保持一致，此处表示在一个Epoch中，采样使用500个nobbps样本，400个bbps0样本，400个bbps1样本，1600个bbps2样本，1600个bbps3样本
--sample_weight_value 500 400 400 1600 1600
# [核心参数] 预处理时缩放图像目标规格，此处表示缩放到(H,W)=(336,336)
--resize_shape 336 336
# [核心参数] 预处理时中心裁剪图像目标规格，此处表示缩放到(H,W)=(336,336)；必须与网络输入端规格匹配，例如ViT-Patch14-336主干网络要求输入规格为(336,336)，ViT-Patch16-224主干网络要求输入规格为(224,224)
# 如果center_crop_shape的某维度与resize_shape不相等，超过的部分被裁剪，不足的部分0填充
--center_crop_shape 336 336
# [非核心参数，保持默认即可] 依次是亮度、对比度、饱和度随机泛化区间，用于缓解过拟合
--brightness_jitter 0.8
--contrast_jitter 0.8
--saturation_jitter 0.8
# 数据装载线程数，可参考实际CPU核心线程数量设定
--num_workers 16
# [核心参数] 网络模型LightningModule类路径，从以下两类中择一
# MultiLabelClassifier.Modelv3中的MultiLabelClassifier_ViT_L_Patch14_336_Class7是使用ViT-Patch14-336主干网络的高分辨率大模型
# MultiLabelClassifier.Modelv2中的MultiLabelClassifier_ViT_L_Patch16_224_Class7是使用ViT-Patch16-224主干网络的低分辨率小模型
--model_class_path MultiLabelClassifier.Modelv3.MultiLabelClassifier_ViT_L_Patch14_336_Class7
# [非核心参数，保持默认即可] Class-Specific Residual Attention(CSRA)输出头数量
--num_heads 8
# [非核心参数，保持默认即可] CSRA权重
--attention_lambda 0.3
# 逐类标签置信度阈值(后处理)，0.5表示当某标签（例如回盲部）置信度>0.5时，才认为此标签存在；此阈值越高对标签特征要求越严格
--thresh 0.5
# [非核心参数，保持默认即可] SGD优化器参数
--lr 0.0001
--momentum 0.9
--weight_decay 0.0001
# 清洁度分类增强损失权重，此权重越大，对清洁度分类的监督越强（另一项损失是权重为1.0的逐标签二元交叉熵损失）
--cls_weight 0.2
# [非核心参数，保持默认即可] 有效模型阈值（模型筛选用），此处表示只有"体外识别准确率>0.9"且"坏帧识别准确率>0.9"的模型才被视作有效模型进行记录
--outside_acc_thresh 0.9
--nonsense_acc_thresh 0.9
# [核心参数] test模式，测试输出时所使用的数据集索引文件，使用其中的idmap（图像标识码-路径映射表），置空时输出模型输入图像（预处理后的），有效时输出索引到的原始图像
--test_id_map_file_path ../Datasets/UIHNJMuLv3/Manifest.json
# [核心参数] test模式，分类错误图像的保存目录，置空时不保存
--test_viz_save_dir Experiment/R103_train_vitp14s336c7_400/test_viz
# 重定向输出日志路径
> log/R103_train_vitp14s336c7_400.log
```

Detailed usage:

启动命令参数详细用法：

```bash
usage: QuickLauncher.py [-h] -s 
                        {fit,finetune,validate,test,predict,export_model_torch_script,arg_debug}
                        [-cm] [-msp MODEL_SAVE_PATH] [-psp PRED_SAVE_PATH]
                        [-se SEED_EVERYTHING] [-me MAX_EPOCHS]
                        [-bs BATCH_SIZE] [-cp CKPT_PATH]
                        [-acc {cpu,gpu,tpu,ipu,auto}]
                        [-str {ddp,ddp_spawn,ddp_notebook}]
                        [-dev DEVICES [DEVICES ...]]
                        [-cve CHECK_VAL_EVERY_N_EPOCH] [-ls LOG_EVERY_N_STEPS]
                        [-en EXPERIMENT_NAME] [-ver VERSION]
                        [-ce CKPT_EVERY_N_EPOCHS] [-trr TQDM_REFRESH_RATE]
                        [-dcp DATA_CLASS_PATH] [-dif DATA_INDEX_FILE]
                        [-dr DATA_ROOT]
                        [-swk SAMPLE_WEIGHT_KEY [SAMPLE_WEIGHT_KEY ...]]
                        [-swv SAMPLE_WEIGHT_VALUE [SAMPLE_WEIGHT_VALUE ...]]
                        [-rs RESIZE_SHAPE RESIZE_SHAPE]
                        [-ccs CENTER_CROP_SHAPE CENTER_CROP_SHAPE]
                        [-bj BRIGHTNESS_JITTER] [-cj CONTRAST_JITTER]
                        [-sj SATURATION_JITTER] [-nw NUM_WORKERS]
                        [-mcp MODEL_CLASS_PATH] [-nh {1,2,4,6,8}]
                        [-al ATTENTION_LAMBDA] [-thr THRESH [THRESH ...]]
                        [-lr LR] [-mom MOMENTUM] [-wd WEIGHT_DECAY]
                        [-cw CLS_WEIGHT] [-oat OUTSIDE_ACC_THRESH]
                        [-nat NONSENSE_ACC_THRESH]
                        [-timfp TEST_ID_MAP_FILE_PATH]
                        [-tvsd TEST_VIZ_SAVE_DIR]

肠镜多任务质控启动器

options:
  -h, --help            show this help message and exit
  -s {fit,finetune,validate,test,predict,export_model_torch_script,arg_debug}, --stage {fit,finetune,validate,test,predict,export_model_torch_script,arg_debug}
                        运行模式：fit-训练(包含训练时验证，检查点用于恢复状态)，finetune-
                        优化（检查点用于重启训练），validate-验证，test-测试，predict-
                        预测，export_model_torch_script-
                        导出TorchScript模型，arg_debug-仅检查参数 (default: None)
  -cm, --compile_model  编译模型以加速(使用GPU，要求CUDA Compute Capability >= 7.0)
                        (default: False)
  -msp MODEL_SAVE_PATH, --model_save_path MODEL_SAVE_PATH
                        TorchScript导出路径，置空时不导出 (default: None)
  -psp PRED_SAVE_PATH, --pred_save_path PRED_SAVE_PATH
                        预测结果保存路径，置空时不保存 (default: None)
  -se SEED_EVERYTHING, --seed_everything SEED_EVERYTHING
                        随机种子 (default: 0)
  -me MAX_EPOCHS, --max_epochs MAX_EPOCHS
                        训练纪元总数 (default: 400)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        批大小 (default: 16)
  -cp CKPT_PATH, --ckpt_path CKPT_PATH
                        模型检查点路径，置空时不装载 (default: None)
  -acc {cpu,gpu,tpu,ipu,auto}, --accelerator {cpu,gpu,tpu,ipu,auto}
                        加速器 (default: gpu)
  -str {ddp,ddp_spawn,ddp_notebook}, --strategy {ddp,ddp_spawn,ddp_notebook}
                        运行策略 (default: ddp)
  -dev DEVICES [DEVICES ...], --devices DEVICES [DEVICES ...]
                        设备号 (default: [0, 1, 2, 3])
  -cve CHECK_VAL_EVERY_N_EPOCH, --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        验证纪元间隔，1表示每个训练纪元运行一次验证流程 (default: 1)
  -ls LOG_EVERY_N_STEPS, --log_every_n_steps LOG_EVERY_N_STEPS
                        日志记录间隔，1表示每个迭代轮次记录一次日志 (default: 10)
  -en EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        实验名称，用于生成实验目录 (default: R001_Releasev1_train_MultiLabe
                        lClassifier_ViT_L_patch16_224_compile_epoch1000)
  -ver VERSION, --version VERSION
                        实验版本号 (default: v1)
  -ce CKPT_EVERY_N_EPOCHS, --ckpt_every_n_epochs CKPT_EVERY_N_EPOCHS
                        检查点保存间隔，1表示每个训练纪元保存一次检查点 (default: 20)
  -trr TQDM_REFRESH_RATE, --tqdm_refresh_rate TQDM_REFRESH_RATE
                        进度条刷新间隔，1表示每个迭代轮次进行一次刷新 (default: 20)
  -dcp DATA_CLASS_PATH, --data_class_path DATA_CLASS_PATH
                        数据模型类路径 (default: ColonoscopyMultiLabelDataModule)
  -dif DATA_INDEX_FILE, --data_index_file DATA_INDEX_FILE
                        数据集索引文件 (default:
                        Datasets/UIHNJMuLv3/folds/fold0.json)
  -dr DATA_ROOT, --data_root DATA_ROOT
                        数据集根路径 (default: Datasets/UIHNJMuLv3)
  -swk SAMPLE_WEIGHT_KEY [SAMPLE_WEIGHT_KEY ...], --sample_weight_key SAMPLE_WEIGHT_KEY [SAMPLE_WEIGHT_KEY ...]
                        重采样数据子集列表 (default: ['ileocecal', 'nofeature',
                        'nonsense', 'outside'])
  -swv SAMPLE_WEIGHT_VALUE [SAMPLE_WEIGHT_VALUE ...], --sample_weight_value SAMPLE_WEIGHT_VALUE [SAMPLE_WEIGHT_VALUE ...]
                        重采样数量列表(与sample_weight_key一一对应) (default: [4800, 4800,
                        480, 96])
  -rs RESIZE_SHAPE RESIZE_SHAPE, --resize_shape RESIZE_SHAPE RESIZE_SHAPE
                        预处理时缩放图像目标规格；格式：(H, W) (default: (224, 224))
  -ccs CENTER_CROP_SHAPE CENTER_CROP_SHAPE, --center_crop_shape CENTER_CROP_SHAPE CENTER_CROP_SHAPE
                        中心裁剪规格，配合resize_shape使用可裁去边缘；格式：(H, W)（注：只有中心(H,
                        W)区域进入网络，以匹配主干网络的输入规格） (default: (224, 224))
  -bj BRIGHTNESS_JITTER, --brightness_jitter BRIGHTNESS_JITTER
                        标准化亮度泛化域宽，[max(0, 1 - brightness), 1 + brightness]
                        (default: 0.8)
  -cj CONTRAST_JITTER, --contrast_jitter CONTRAST_JITTER
                        标准化对比度泛化域宽，[max(0, 1 - contrast), 1 + contrast]
                        (default: 0.8)
  -sj SATURATION_JITTER, --saturation_jitter SATURATION_JITTER
                        标准化饱和度泛化域宽，[max(0, 1 - saturation), 1 + saturation]
                        (default: 0.8)
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        数据装载线程数 (default: 16)
  -mcp MODEL_CLASS_PATH, --model_class_path MODEL_CLASS_PATH
                        网络模型类路径 (default:
                        MultiLabelClassifier_ViT_L_Patch16_224_Class7)
  -nh {1,2,4,6,8}, --num_heads {1,2,4,6,8}
                        输出头（不同温度T）数量 (default: 8)
  -al ATTENTION_LAMBDA, --attention_lambda ATTENTION_LAMBDA
                        输出头类特征权重 (default: 0.3)
  -thr THRESH [THRESH ...], --thresh THRESH [THRESH ...]
                        逐类标签置信度阈值 (default: [0.5])
  -lr LR, --lr LR       SGD优化器学习率 (default: 0.0001)
  -mom MOMENTUM, --momentum MOMENTUM
                        SGD优化器动量 (default: 0.9)
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        SGD优化器权重衰退 (default: 0.0001)
  -cw CLS_WEIGHT, --cls_weight CLS_WEIGHT
                        清洁度损失权重 (default: 0.2)
  -oat OUTSIDE_ACC_THRESH, --outside_acc_thresh OUTSIDE_ACC_THRESH
                        outside性能筛选线 (default: 0.9)
  -nat NONSENSE_ACC_THRESH, --nonsense_acc_thresh NONSENSE_ACC_THRESH
                        nonsense性能筛选线 (default: 0.9)
  -timfp TEST_ID_MAP_FILE_PATH, --test_id_map_file_path TEST_ID_MAP_FILE_PATH
                        测试输出时所使用的数据集索引文件，使用其中的图像标识码-
                        路径映射表，置空时输出模型输入图像，有效时输出索引到的原始图像 (default: None)
  -tvsd TEST_VIZ_SAVE_DIR, --test_viz_save_dir TEST_VIZ_SAVE_DIR
                        测试时，分类错误图像的保存目录，置空时不保存 (default: None)
```

## Launch with Deployment Configuration (Alternative)

## 从配置文件启动（备用启动方式）

Example config YAML for training and testing can be found in:

训练、测试用示例YAML配置文件可参见如下目录：

```bash
<ProjectRoot>/Config/
```

Start training from config:

使用如下命令从配置文件启动训练：

注：从配置文件启动方式下无法启用PyTorch2.0的编译加速功能

```bash
python DeployLauncher.py fit --config <ProjectRoot>/Config/<ConfigFile.yaml>
```

Lightning CLI Configuration References:

Lightning通过配置文件启动: 

[Configure hyperparameters from the CLI (Intermediate)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html)

[Configure hyperparameters from the CLI (Advanced)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html)

## Logging

## 日志

Logs are in both forms:

日志包含两种格式：

```
tensorboard
csv
```

Log files save at:

快速启动时，日志文件保存于重定向目标位置，无重定向时仅输出到控制台。

从配置文件启动时，日志文件保存于：

```bash
<ProjectRoot>/Experiment/<ExperimentName>/
```

## References

## 参考

Also please refer to the Docs: 

另请参考在线资料：

PyTorch Doc: [PyTorch documentation](https://pytorch.org/docs/2.0/)

PyTorch可复现性: [Reproducibility](https://pytorch.org/docs/master/notes/randomness.html#reproducibility)

PyTorch2.0编译相关: [TorchDynamo Troubleshooting](https://pytorch.org/docs/stable/dynamo/troubleshooting.html)

Lightning框架总览：[Lightning in 15 minutes](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

Lightning框架代码组织：[How to Organize PyTorch Into Lightning](https://lightning.ai/docs/pytorch/stable/starter/converting.html)

Lightning通过配置文件启动：

[Configure hyperparameters from the CLI (Intermediate)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html)

[Configure hyperparameters from the CLI (Advanced)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html)

通过Lightning使用PyTorch2.0编译：[Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)

