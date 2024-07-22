import glob, os
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import load, device

from .config import Config
from .trainer import TrainerDDPM, TrainerSupervised, TrainerBase

# Import all models from files in ../models/*
DEFAULT_MODEL_PATH = os.path.dirname(__file__).replace("utils", "models")
module_path = DEFAULT_MODEL_PATH.replace(os.getcwd(), "").replace("/", ".")[1:]
for p in glob.glob(os.path.join(DEFAULT_MODEL_PATH, "*.py")):
    filename = os.path.split(p)[1].replace(".py", "")
    exec(f"from {module_path}.{filename} import *")


class ModelLoader:
    """
    Load model fron model configuration file
    """
    def __init__(self, model_name, cfg_model) -> None:
        self.model = None
        self.model_name = model_name
        self.cfg_model = cfg_model

    def get_model(self) -> Module:
        try:
            exec(f"self.model = {self.model_name}(**self.cfg_model)")
            self.model.__setattr__("name", self.model_name)
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model number of trainable parameters:", n_params)
        except Exception as e:
            print(e)
            print("Can't load model", self.model_name)
        
        return self.model
    
def get_model(cfg:Config=None, state_path:str="") -> Module:
    """
    Return model assuming it is save in logs with its config yaml file
    """
    model = None
    if state_path != "":
        # Get run config
        run_dir = os.path.split(state_path)[0]
        config_path = glob.glob(run_dir + "/*.yaml") + glob.glob(run_dir + "/*.yml")
        cfg = Config(config_path[0])

        # Load model
        cfg_model = {**cfg.model["PARAMS"]}
        m = ModelLoader(cfg.get_value("model_name"), cfg_model)
        model = m.get_model()
        state = load(state_path, map_location=device('cpu'))
        model.load_state_dict(state["state_dict"])
        print("Model state restored at epoch", state["epoch"])
        return model
    
    elif cfg != None:
        model_loader = ModelLoader(cfg.get_value("model_name"), cfg.model['PARAMS'])
        model = model_loader.get_model()

    return model

def get_config(state_path:str="") -> Config:
    # Get run config
    run_dir = os.path.split(state_path)[0]
    config_path = glob.glob(run_dir + "/*.yaml") + glob.glob(run_dir + "/*.yml")
    cfg = Config(config_path[0])
    
    return cfg
