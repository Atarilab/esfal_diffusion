from .dataset_DDPM import get_dataloaders as get_dataloaders_DDPM
from .dataset_supervised import get_dataloaders as get_dataloaders_supervised
from .dataset_CDCD import get_dataloaders as get_dataloaders_CDCD

class DataloaderFactory():
    DATALOADERS = {
        "supervised": get_dataloaders_supervised,
        "ddpm": get_dataloaders_DDPM,
        "cdcd": get_dataloaders_CDCD
    }

    @staticmethod
    def get_dataloaders(training_mode:str, params:dict):
        dataloader_factory = DataloaderFactory.DATALOADERS.get(training_mode.lower())
        if not dataloader_factory:
            raise ValueError("Invalid training mode. Should be supervised or ddpm.")
        return dataloader_factory(**params)

