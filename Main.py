# main.py
import torch.backends.cudnn
from lightning import seed_everything
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from Classifier import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()

    # python Main.py fit --config Config/site_quality_config/config_site_quality_resnet50_train.yaml
    #
