import tyro
import os

from tree_search.experiment_manager import ExperimentManager

def main(
    record_dir : str = '../data',
    N_runs : int = 300,
    N_stones_removed : int = 9,
    N_goal_per_env : int = 1,
    N_sol_per_goal : int = 5,
    N_repeat_per_sol : int = 5,
    stepping_stones_height :float = 0.1,
    randomize_height_ratio :float = 0.25,
    randomize_pos_ratio :float = 0.45,
    mcts_C :float = 1.e-2,
    mcts_W :float = 1,
    mcts_alpha_exploration :float = 0.,
    mcts_max_step_size: float = 0.23,
    size_ratio : float = 0.6,
    gait : str = 'jump',
    n_cores : int = 8,
    load_experiment : str = ""
    ):
    
    if load_experiment != "":
        manager = ExperimentManager.load(load_experiment)
    else:
        manager = ExperimentManager(**locals())
        
    manager.start(n_cores)
    
if __name__ == "__main__":
    args = tyro.cli(main)