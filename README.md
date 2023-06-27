# Colonoscope Quality Control

It is an implementation based on PyTorch and Lightning.

实现基于PyTorch和Lightning框架。

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

## 环境配置

使用Anaconda配置运行环境

创建虚拟环境`<VirtualEnv>`，安装Python3.10解释器：

```bash
conda create -n <VirtualEnv> python=3.10 -y
```

进入环境：

```bash
conda activate cwypy310pt20
```

安装PyTorch2.0相关组件：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

安装Lightning相关组件：

```bash
conda install lightning "pytorch-lightning[extra]" -c conda-forge -y
```

安装模型库、日志记录器等辅助支持：

```bash
conda install tensorboardX timm -c conda-forge -y
```

安装实用库：

```bash
conda install numpy pyyaml tqdm pillow
```

## 启动

Entry of the program is in:

程序入口位于：

```bash
<ProjectRoot>/QuickLauncher.py
```

You may use this for a quick training start:

使用如下命令启动训练：

```bash
python QuickLauncher.py --stage fit --device <GPU-NumberList, eg: 0,1>
```

## 日志

Logs are in both forms:

日志包含两种格式：

```
tensorboard
csv
```
Log files are saved at:

日志文件保存于：

```bash
<ProjectRoot>/Experiment/<ExperimentName>/
```

## 从配置文件启动

Example config YAML for training and testing can be found in:

训练、测试用示例YAML配置文件可参见如下目录：

```bash
<ProjectRoot>/Config/
```

Start training from config:

使用如下命令从配置文件启动训练：

```bash
python DeployLauncher.py fit --config <ProjectRoot>/Config/<ConfigFile.yaml>
```

## 参考

Also please refer to the Docs: 

另请参考在线资料：

PyTorch Doc: [PyTorch documentation — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/)

PyTorch可复现性: [Reproducibility — PyTorch master documentation](https://pytorch.org/docs/master/notes/randomness.html#reproducibility)

PyTorch2.0编译相关: [TorchDynamo Troubleshooting — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/dynamo/troubleshooting.html)

Lightning框架总览: [Lightning in 15 minutes — PyTorch Lightning 2.0.4 documentation](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

Lightning框架代码组织: [How to Organize PyTorch Into Lightning — PyTorch Lightning 2.0.4 documentation](https://lightning.ai/docs/pytorch/stable/starter/converting.html)

Lightning通过配置文件启动: [Configure hyperparameters from the CLI (Intermediate) — PyTorch Lightning 2.0.4 documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html)

通过Lightning使用PyTorch2.0编译: [Training Compiled PyTorch 2.0 with PyTorch Lightning](https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/)

