

import tyro
import glob
import os
import shutil
from typing import Optional

from data.utils import DataloaderFactory
from utils.config import Config
from utils.sweep import SweepParametersConfig
from utils.trainer import TrainerFactory
from utils.utils import get_model, DEFAULT_MODEL_PATH, DEFAULT_MODELS_DIR

SUPERVISED_MODE = "supervised"
DDPM_MODE = "ddpm"
DEFAULT_CONFIG_PATH = "./config/supervised.yml"

def main(
    train: str = SUPERVISED_MODE,
    test: bool = False,
    cfg: str = DEFAULT_CONFIG_PATH,
    model_path: Optional[str] = "",
    run_desc: Optional[str] = "",
    ) -> None:
    """
    Args:
        - train: train model. Specify which training mode to use: supervised (default), ddpm
        - test: test model.
        - cfg: configuration file path (.yml).
        - model_path: model path to load (.pth). Load corresponding config.
        - run_desc: Name given to the run dir. Usefull to give a short description of the run.
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
        print("Using config file from", rundir)
        cfg.change_value("logdir", rundir)

    ### DATA
    dataloader_train, dataloader_test = DataloaderFactory.get_dataloaders(train, params=cfg.data)

    ### TRAIN
    if train:

        sweeper = SweepParametersConfig(cfg, cfg.sweep)

        for cfg in sweeper:

            ### MODEL
            model = get_model(cfg, state_path=model_path)

            trainer = TrainerFactory.create_trainer(train)
            trainer = trainer(cfg, model, dataloader_train, dataloader_test, run_dir_name=run_desc)
            
            # Copy models files to run directory
            rel_models_path = os.path.join(trainer.run_dir, DEFAULT_MODELS_DIR)
            if not os.path.exists(rel_models_path):
                shutil.copytree(DEFAULT_MODEL_PATH, rel_models_path)

            # Train
            trainer.train()

    if test and model_path:
        ### MODEL
        model = get_model(cfg, state_path=model_path)
        
        model.eval()
        # Do what you want

if __name__ == "__main__":
    args = tyro.cli(main)

