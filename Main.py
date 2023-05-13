# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from Classifier import *


def cli_main():
    cli = LightningCLI()
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()

    # python Main.py fit --config Config/site_quality_config/config_site_quality_resnet50_train.yaml
    #
