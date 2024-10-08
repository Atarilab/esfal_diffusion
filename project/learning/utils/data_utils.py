import os
import glob
import importlib.util
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DEFAULT_DATA_DIR = "data"
DEFAULT_DATASET_PATH = os.path.dirname(__file__).replace("utils", DEFAULT_DATA_DIR)
DEFAULT_SPLIT_RATIO = 0.2
DEFAULT_BATCH_SIZE = 128

class DatasetLoader:
    """
    Load datasets dynamically from dataset files in the specified directory.
    """
    def __init__(self, absolute_data_path: str = "") -> None:
        self.dataset = None
        # Set the path for dataset files
        absolute_data_path = DEFAULT_DATASET_PATH if absolute_data_path == "" else absolute_data_path
        self.all_datasets = {}
        # Import datasets from files
        self._import_datasets(absolute_data_path)

    def _import_datasets(self, absolute_data_path: str):
        # Create relative path
        module_path = absolute_data_path.replace(os.getcwd(), "")
        # Parse as python file import
        module_path = module_path.replace("/", ".")[1:]
        for p in glob.glob(os.path.join(absolute_data_path, "*.py")):
            filename = os.path.split(p)[1].replace(".py", "")

            # Import datasets
            module_name = f"{module_path}.{filename}"
            spec = importlib.util.spec_from_file_location(module_name, p)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get all the classes that inherit from Dataset defined in the module
            self.all_datasets.update(
                {name: cls for name, cls in module.__dict__.items()
                 if isinstance(cls, type) and issubclass(cls, Dataset)}
            )

    def load(self, dataset_name: str, **kwargs) -> Dataset:
        try:
            # Load dataset class by name and instantiate it with provided arguments
            self.dataset = self.all_datasets[dataset_name](**kwargs)
            print(f"Dataset {dataset_name} loaded successfully.")
        except Exception as e:
            print(e)
            print(f"Can't load dataset {dataset_name}")
        
        return self.dataset

def get_dataloaders(dataset_name: str, cfg_dataset: dict) -> Tuple[DataLoader, DataLoader, str]:
    """
    Create train and validation DataLoaders from the given dataset.
    The last `split_ratio` of the dataset is used for validation.

    Args:
        dataset (Dataset): The dataset to be split and loaded.
        batch_size (int): The batch size for the DataLoaders.
        split_ratio (float): The ratio of the dataset to be used for validation.
        dataloader_kwargs (dict): Additional arguments to pass to the DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: Train DataLoader and Validation DataLoader.
    """
    
    # Get dataset from name and config
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.load(dataset_name, **cfg_dataset)
    normalization_file_path = ""
    if hasattr(dataset, "normalization_file_path"):
        normalization_file_path = dataset.normalization_file_path
    
    # Split dataset
    split_ratio = cfg_dataset.get("split_ratio", DEFAULT_SPLIT_RATIO)
    total_size = len(dataset)
    val_size = int(total_size * split_ratio)
    train_size = total_size - val_size
    manual_seed = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], manual_seed)

    # Create DataLoaders
    batch_size = cfg_dataset.get("batch_size", DEFAULT_BATCH_SIZE)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Print info
    print("Number of samples:")
    print("Train:", len(train_dataset))
    print("Test:", len(val_dataset))
    print()

    print("Train batch shape:")
    batch = next(iter(train_loader))
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_loader, val_loader, normalization_file_path

