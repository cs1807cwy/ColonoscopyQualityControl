# Colonoscope Quality Control

It is an implementation based on PyTorch and PyTorch Lightning.

Already tested on:
```
python 3.10.10
pytorch 2.0.1
pytorch-lightning 2.0.2
GPU Nvidia GeForce GTX 1080ti
```

Entry of the program is in:
```
<ProjectRoot>/Main.py
```

You may use this for a quick start:
```
python Main.py fit --config "Config/config_train_example.yaml"
```

Logs are in both forms:
```
tensorboard
csv
```
Log files are saved in: (you may modify this via config YAML, see below)
```
<ProjectRoot>/<TaskConfig>/Experiment/
```

Example config YAML for training and testing can be found in:
```
<ProjectRoot>/Config/<TaskConfig>
```

Many key configurations are detailly noted in:
```
<ProjectRoot>/Config/<TaskConfig>/config_train_example.yaml

```
Also please refer to the Docs: [Pytorch Lighting 2.0.2](https://pytorch-lightning.readthedocs.io/en/2.0.2/starter/introduction.html)