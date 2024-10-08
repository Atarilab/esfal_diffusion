

import tyro
import glob
import os
import shutil
from typing import Optional

from utils.config import Config
from utils.sweep import SweepParametersConfig
from utils.trainer import TrainerBase, TrainerFactory
from utils.model_utils import get_model, get_model_and_config
from utils.data_utils import get_dataloaders


def main(
    cfg: Optional[str] = "",
    model_path: Optional[str] = "",
    desc: Optional[str] = "",
    ) -> None:
    """
    Args:
        - cfg: configuration file path (.yml).
        - model_path: model path to load (.pth). Load corresponding config.
        - desc: Description given to the run dir. Usefull to give a short description of the run.
    """
    ### CONFIG

    # Load pretrain model
    # Load config from config file in run directory
    if model_path != "" and cfg == "":
        model, config = get_model_and_config(model_path)
    else:
        config = Config(cfg)

    ### DATA
    dataloader_train, dataloader_test, normalization_file = get_dataloaders(*config.data())

    ### TRAIN
    for config in SweepParametersConfig(config):

        model = get_model(config, state_path=model_path)
        
        trainer_ = TrainerFactory.get_trainer(config.trainer()[0])
        trainer : TrainerBase = trainer_(config, model, dataloader_train, dataloader_test, run_dir_name=desc)
        trainer.copy_normalization_stats_to_run_dir(normalization_file)
        trainer.train()

if __name__ == "__main__":
    args = tyro.cli(main)

