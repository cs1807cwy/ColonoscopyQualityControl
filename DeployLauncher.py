# main.py
import torch.backends.cudnn
from lightning import seed_everything
from lightning.pytorch.cli import LightningCLI

# for reproduction
# 用于复现
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    # note: don't call fit directly within code!! You should use command line parameter 'fit', 'validate', 'test', 'predict'


if __name__ == "__main__":
    cli_main()
    # python Main.py fit --config /path/to/Config.yaml
