import tyro
import glob
import os

from typing import Optional
from utils.config import Config
from utils.sweep import SweepParametersConfig
from utils.trainer import TrainerDDPM, TrainerSupervised
from utils.utils import get_model

SUPERVISED_MODE = "supervised"
DDPM_MODE = "ddpm"

def main(
    train: str = "supervised",
    test: bool = False,
    cfg: str = "./config/supervised.yml",
    dataset_dir: str = "",
    model_path: Optional[str] = ""
    ) -> None:
    """
    Args:
        train: train model. Specify which training mode to use: supervised (default), ddpm
        test: test model.
        cfg: configuration file path (.yml).
        dataset_dir: dataset directory path.
        model_path: model path to load (.pth). Load corresponding config.
    """
    ### CONFIG
    if cfg != "":
        cfg = Config(cfg)

    # Load config from config file in run directory
    if model_path != "":
        rundir = os.path.split(model_path)[0]
        cfg_run = glob.glob(rundir + "/*.yaml") + glob.glob(rundir + "/*.yml")
        if len(cfg_run) > 0:
            cfg = Config(cfg_run[0])
        cfg.change_value("logdir", rundir)

    ### DATA
    if dataset_dir != "":
        cfg.change_value("data_dir", dataset_dir)
    if train.lower() == SUPERVISED_MODE:
        from data.dataset_supervised import get_dataloaders
    elif train.lower() == DDPM_MODE:
        from data.dataset_DDPM import get_dataloaders # A bit dirty

    dataloader_train, dataloader_test = get_dataloaders(**cfg.data)
    
    ### MODEL
    model = get_model(cfg, state_path=model_path)

    ### TRAIN
    if train:

        sweeper = SweepParametersConfig(cfg, cfg.sweep)

        for cfg in sweeper:

            trainer = None
            if train.lower() == SUPERVISED_MODE:
                trainer = TrainerSupervised(cfg, model, dataloader_train, dataloader_test)
            elif train.lower() == DDPM_MODE:
                trainer = TrainerDDPM(cfg, model, dataloader_train, dataloader_test)

            trainer.train()

    if test:

        model = model.eval()
        # Do what you want

if __name__ == "__main__":
    args = tyro.cli(main)
