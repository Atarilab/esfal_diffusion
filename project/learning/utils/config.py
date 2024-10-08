import yaml
import os
import shutil
from typing import List, Tuple

class Config():

    DATA_KEY = "data"
    MODEL_KEY = "model"
    TRAINER_KEY = "trainer"
    SWEEP_KEY = "sweep"
    
    VALID_KEYS = [
        DATA_KEY,
        MODEL_KEY,
        TRAINER_KEY,
        SWEEP_KEY,
    ]

    def __init__(self, config_path) -> None:
        self.config_path = config_path
        self.data_cfg = self._load_parameters_from_yaml()

    def _load_parameters_from_yaml(self):
        with open(self.config_path, 'r') as file:
            parameters = yaml.safe_load(file)
        return parameters
    
    def get_value(self, parameter_str:str, d:dict={}):
        """
        Return parameter value from parameter name
        """
        current_dict = self.data_cfg if len(d) == 0 else d
        for key, value in current_dict.items():
            if isinstance(key, str) and key.lower() == parameter_str.lower():
                return value
            if type(value) == dict:
                res = self.get_value(parameter_str, value)
                if res != None:
                    return res
        return None

    def change_value(self, parameter_str:str, new_value, d:dict={}) -> bool:
        """
        Change value of a config parameter.
        """
        current_dict = self.data_cfg if len(d) == 0 else d
        for key, value in current_dict.items():
            if type(value) == dict:
                self.change_value(parameter_str, new_value, value)
            if key == parameter_str:
                current_dict[key] = new_value
                return True
        return False
    
    def _get_subdict_if_exists(self, key_list: List[str]):
        """
        Returns a subdict of config dict with key in <key_list> (to manage typing errors).
        """
        d = {}
        for model_key in key_list:
            if model_key in self.data_cfg:
                d = self.data_cfg[model_key]
        return d
    
    def get_cfg_as_dict(self, d:dict={}):
        """
        Returns config as a flatten dict.
        """
        flatten_d = {}
        current_dict = self.data_cfg if len(d) == 0 else d
        for key, value in current_dict.items():
            if type(value) == dict:
                flatten_d = {**flatten_d, **self.get_cfg_as_dict(value)}
            else:
                flatten_d[key] = value
        return flatten_d
    
    def write(self, path:str):
        """
        Write config data in a yaml file.
        """
        if path[-5:] != ".yaml" and path[-4:] != ".yml":
            path = os.path.join(path, "config.yaml")

        shutil.copy(self.config_path, path)
    
    def _get_name_params(self, cfg_key : str) -> Tuple[str, dict]:
        
        if not cfg_key in Config.VALID_KEYS:
            raise ValueError(f"Config key '{cfg_key}' not valid.")
        
        cfg_full = self.get_value(cfg_key)
        if cfg_full is None:
            name = ""
            cfg_params = {}
        else:
            name = cfg_full.get("name", "")
            if name:
                cfg_params = cfg_full["PARAMS"]
            else:
                cfg_params = cfg_full
            
        return name, cfg_params
        
    def data(self) -> Tuple[str, dict]:
        return self._get_name_params(Config.DATA_KEY)
    
    def trainer(self) -> Tuple[str, dict]:
        return self._get_name_params(Config.TRAINER_KEY)
    
    def model(self) -> Tuple[str, dict]:
        return self._get_name_params(Config.MODEL_KEY)
    
    def sweep(self) -> Tuple[str, dict]:
        return self._get_name_params(Config.SWEEP_KEY)