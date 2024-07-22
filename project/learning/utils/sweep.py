import numpy as np
import itertools
import copy
import os

from .config import Config

class SweepParametersConfig():
    def __init__(self,
                 cfg:Config,
                 swp_cfg:dict={}) -> None:
        self.cfg = cfg
        self.swp_cfg = swp_cfg if swp_cfg != None else {}

        self.params, self.param_combinations = self.generate_cartesian_product()
        self.current_combination_id = 0

    def generate_cartesian_product(self):
        param_values = []
        param_name = []
        for param, value in self.swp_cfg.items():
            if self.cfg.get_value(param) != None:
                if isinstance(value, dict):
                    n = 2
                    if "n" in value.keys():
                        n = value["n"]
                    if value.get('logspace'):
                        param_values.append(list(self.python_type(v) for v in np.logspace(np.log10(value['min']), np.log10(value['max']), n)))
                    else:
                        dtype = type(value['min'])
                        param_values.append(list(self.python_type(v) for v in np.linspace(value['min'], value['max'], n, dtype=dtype)))
                elif isinstance(value, list):
                    param_values.append(value)

                param_name.append(param)
            else:
                print(f"Can't find sweep parameter {param}")

        param_combinations = list(itertools.product(*param_values))
        return param_name, param_combinations
    
    def python_type(self, v):
        if int(v) == v:
            return int(v)
        return float(v)
    
    def sweep_run_name(self):
        name = "sweep_"
        for param in self.params:
            name += param + "_"
        name = name[:-1]
        return name
    
    def next_config(self):
        if self.current_combination_id >= len(self.param_combinations):
            return None

        new_cfg = copy.deepcopy(self.cfg)
        if len(self.param_combinations) > 1:
            # Change logdir name
            print(f"\n--- Sweep parameters {self.current_combination_id + 1}/{len(self.param_combinations)}")
            old_logdir = self.cfg.get_value("logdir")
            new_cfg.change_value("logdir", os.path.join(old_logdir, self.sweep_run_name()))

            values = self.param_combinations[self.current_combination_id]
            for (param, value) in zip(self.params, values):
                print("-", param, value)
                new_cfg.change_value(param, value)
        
        self.current_combination_id += 1
        return new_cfg
    
    def __iter__(self):
        return self

    def __next__(self):
        self.current_cfg = self.next_config()

        if self.current_cfg == None:
            raise StopIteration
        
        return self.current_cfg