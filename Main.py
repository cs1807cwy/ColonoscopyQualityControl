# main.py
from lightning.pytorch.cli import LightningCLI
from Classifier.DataModule import ColonoscopySiteQualityDataModule
from Classifier.Model import CQCClassifier
from Classifier.Train import CQCTrainer

# simple demo classes for your convenience
from Classifier import *


def cli_main():
    cli = LightningCLI(model_class=CQCClassifier, datamodule_class=ColonoscopySiteQualityDataModule, trainer_class=CQCTrainer)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block