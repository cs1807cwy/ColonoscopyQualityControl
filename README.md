# Colonoscope Quality Control

Note&Warn: this implementation is produced by a greenhand newly in deep learning, bounty of errors may occur.

It is an implementation based on PyTorch and PyTorch Lightning.

Already tested on:
```
python 3.10.11
pytorch 2.0.1
pytorch-lightning 2.0.2
GPU Nvidia GeForce GTX 1080ti
NVCC V11.7.99
```

Entry of the program is in:
```
<ProjectRoot>/Main.py
```

You may use this for a quick start:
```
python Main.py fit --config "Config/<task_config_folder>/config_train_example.yaml"
```

Logs are in both forms:
```
tensorboard
csv
```
Log files are saved in: (you may modify this via config YAML, see below)
```
<ProjectRoot>/Experiment/
```

Example config YAML for training and testing can be found in:
```
<ProjectRoot>/Config/
```

Many key configurations are detailly noted in:
```
<ProjectRoot>/Config/<task_config_folder>/config_train_example.yaml

```
Also please refer to the Docs: [Pytorch Lightning 2.0.2](https://pytorch-lightning.readthedocs.io/en/2.0.2/starter/introduction.html)