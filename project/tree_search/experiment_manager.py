import os
import glob
import pickle
import time
import mujoco
import numpy as np
import yaml
import tqdm
import multiprocessing
import gc

from environment.simulator import SteppingStonesSimulator
from environment.stepping_stones import SteppingStonesEnv
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from tree_search.mcts_stepping_stones import MCTSSteppingStonesKin
from tree_search.data_recorder import ContactsDataRecorder
from utils.config import Go2Config

class ExperimentManager(object):
    DEFAULT_STEPPING_STONES_HEIGHT = 0.1
    DEFAULT_HEIGHT_RATIO = 0.2
    DEFAULT_POS_RATIO = 0.8
    DEFAULT_MAX_STEP_SIZE = 0.23
    DEFAULT_MAX_IT = 10000
    DEFAULT_SIZE_RATIO = 0.55
    DEFAULT_C = 4.0e-2
    DEFAULT_W = 1.
    DEFAULT_ALPHA_EXPLORATION = 0.
    DEFAULT_SIM_STEP = 1
    DEFAULT_GAIT = "jump"
    CONFIG_FILE_NAME = "experiment_config.yaml"
    
    def __init__(self,
                 N_runs : int,
                 N_stones_removed : int,
                 N_goal_per_env : int,
                 N_sol_per_goal : int,
                 N_repeat_per_sol : int,
                 record_dir : str = "",
                 **kwargs,
                 ) -> None:
        
        self.N_runs = N_runs
        self.N_stones_removed = N_stones_removed
        self.N_goal_per_env = N_goal_per_env
        self.N_sol_per_goal = N_sol_per_goal
        self.N_repeat_per_sol = N_repeat_per_sol
        
        optionals_parameters = {
            "stepping_stones_height" : ExperimentManager.DEFAULT_STEPPING_STONES_HEIGHT,
            "randomize_height_ratio" : ExperimentManager.DEFAULT_HEIGHT_RATIO,
            "randomize_pos_ratio" : ExperimentManager.DEFAULT_POS_RATIO,
            "size_ratio" : ExperimentManager.DEFAULT_SIZE_RATIO,
            "mcts_C" : ExperimentManager.DEFAULT_C,
            "mcts_W" : ExperimentManager.DEFAULT_W,
            "mcts_alpha_exploration" : ExperimentManager.DEFAULT_ALPHA_EXPLORATION,
            "mcts_sim_step" : ExperimentManager.DEFAULT_SIM_STEP,
            "mcts_max_step_size" : ExperimentManager.DEFAULT_MAX_STEP_SIZE,
            "mcts_n_it" : ExperimentManager.DEFAULT_MAX_IT,
            "gait" : ExperimentManager.DEFAULT_GAIT,
        }
        optionals_parameters.update(kwargs)
        for k, v in optionals_parameters.items(): setattr(self, k, v)
        
        self.experiment_dir = ""
        if record_dir != "":
            date_time_str = time.strftime('%d%m-%H%M%S')
            self.experiment_dir = os.path.join(record_dir, date_time_str)
            os.makedirs(self.experiment_dir, exist_ok=True)
            self.save_experiment_parameters(self.experiment_dir)
        
    def get_stones_env(self) -> SteppingStonesEnv:
        stones_env = SteppingStonesEnv(
            spacing=(0.19, 0.13),
            height=self.stepping_stones_height,
            randomize_height_ratio=self.randomize_height_ratio,
            randomize_pos_ratio=self.randomize_pos_ratio,
            size_ratio=(self.size_ratio, self.size_ratio),
            N_to_remove=self.N_stones_removed
            )
        return stones_env

    def get_mcts(self, robot_paths : list[str], stones_env : SteppingStonesEnv) -> MCTSSteppingStonesKin:
        """
        Setup a mcts run.
        """
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
            *robot_paths,
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
        )

        controller = BiConMPC(robot.pin, height_offset=self.stepping_stones_height)
        gait = trot if self.gait == "trot" else jump
        controller.set_gait_params(gait)
        
        data_recorder = ContactsDataRecorder(
            robot.mj,
            stones_env,
            record_dir=self.experiment_dir
            )
        
        sim = SteppingStonesSimulator(
            stepping_stones_env=stones_env,
            robot=robot.mj,
            controller=controller,
            data_recorder=data_recorder
            )
        
        mcts = MCTSSteppingStonesKin(
            sim,
            simulation_steps=self.mcts_sim_step,
            C=self.mcts_C,
            W=self.mcts_W,
            alpha_exploration=self.mcts_alpha_exploration,
            max_step_size=self.mcts_max_step_size,
            max_solution_search=self.N_sol_per_goal,
            )
        
        return mcts

    def run_single_experiment(self, robot_paths : list, i_env : int):
        """
        Run mcts on a given stepping stones environment.
        - Run MCTS for <N_goal_per_env>.
        - Rerun each plan found <N_repeat_per_sol> with randomized initial states.
        """
        
        stones_env = self.get_stones_env()
        
        env_name = f"env_{i_env}"
        env_dir = os.path.join(self.experiment_dir, env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        ### Set randomized goal
        for i_goal in range(self.N_goal_per_env):
            np.random.seed(i_env + i_goal + int(time.time() % 0.1 * 1111 * (i_env + 10 * (i_goal + 1))))
            
            mcts = self.get_mcts(robot_paths, stones_env)
            mcts.sim.set_start_and_goal()
                
            # Save stones once some are removed
            if i_goal == 0:
                # Save the stepping stones environment
                stones_env.save(env_dir)
            
            goal_name = f"goal_{i_goal}"
            goal_dir = os.path.join(env_dir, goal_name)
            os.makedirs(goal_dir, exist_ok=True)
            
            # Set data recorder path to goal dir
            mcts.sim.data_recorder.update_record_dir(goal_dir)
            
            # Run MCTS
            try:
                mcts.search(num_iterations=self.mcts_n_it)
            except Exception as e:
                print(e)
            
            # Save mcts performances
            mcts.save_search_performances(goal_dir)

            ### Record contact plan for new randomize initial state
            for contact_plan in mcts.all_solutions: # Solutions are already recorded
                for _ in range(self.N_repeat_per_sol):
                    mcts.sim.run_contact_plan(contact_plan, randomize=True, real_time=False, verbose=False)
                
        del mcts, stones_env
        gc.collect()
            
    def start(self, n_cores : int = 10):
        """
        Launch experiment on different cores.
        """
        # In case experiment is run in a existing directory. Set i_env accordingly
        N_old_env = len(list(filter(lambda path : os.path.isdir(os.path.join(self.experiment_dir, path)), os.listdir(self.experiment_dir))))
        
        cfg = Go2Config
        robot_paths = *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        
        processes = []
        for i_env in tqdm.trange(self.N_runs):
            p = multiprocessing.Process(target=self.run_single_experiment, args=[robot_paths, i_env + N_old_env])
            processes.append(p)
            p.start()

            # Maintain the number of concurrent processes
            while len(processes) >= n_cores:
                for p in processes:
                    if not p.is_alive():
                        processes.remove(p)
                        break
                time.sleep(0.1)  # Small sleep to avoid busy waiting

        # Join any remaining processes
        for p in processes:
            p.join()
    
    def gather_data_experiment(self):
        all_data = {}
        
        for i_env in range(self.N_runs):
            env_name = f"env_{i_env}"
            env_dir = os.path.join(self.experiment_dir, env_name)
            
            for i_goal in range(self.N_goal_per_env):
                goal_name = f"goal_{i_goal}"
                goal_dir = os.path.join(env_dir, goal_name)
                
                file_path = os.path.join(goal_dir, ContactsDataRecorder.FILE_NAME)
                
                if os.path.isfile(file_path):
                    with np.load(file_path) as data:
                        for key in data:
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].append(data[key])

        # Concatenate data lists into arrays
        for key in all_data:
            all_data[key] = np.concatenate(all_data[key], axis=0)
        
        # Save combined data to a single .npz file
        output_file = os.path.join(self.experiment_dir, ContactsDataRecorder.FILE_NAME)
        np.savez(output_file, **all_data)
    
    def gather_performance_data(self):
        all_performance_data = {}

        for env_name in os.listdir(self.experiment_dir):
            env_dir = os.path.join(self.experiment_dir, env_name)
            if not os.path.isdir(env_dir):
                continue

            for goal_name in os.listdir(env_dir):
                goal_dir = os.path.join(env_dir, goal_name)
                if not os.path.isdir(goal_dir):
                    continue

                file_path = os.path.join(goal_dir, MCTSSteppingStonesKin.MCTS_PERF_FILE_NAME)
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        performance_data = pickle.load(f)
                        for key, value in performance_data.items():
                            if key not in all_performance_data:
                                all_performance_data[key] = []
                            all_performance_data[key].extend(value)

        file_path = os.path.join(self.experiment_dir, MCTSSteppingStonesKin.MCTS_PERF_FILE_NAME)
        
        # Saving a dictionary to a Pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(all_performance_data, f)

    def save_experiment_parameters(self, saving_dir : str):
        config = {
            "N_runs" : self.N_runs,
            "N_stones_removed" : self.N_stones_removed,
            "N_goal_per_env" : self.N_goal_per_env,
            "N_sol_per_goal" : self.N_sol_per_goal,
            "N_repeat_per_sol" : self.N_repeat_per_sol,
            "stepping_stones_height" : self.stepping_stones_height,
            "randomize_height_ratio" : self.randomize_height_ratio,
            "randomize_pos_ratio" : self.randomize_pos_ratio,
            "mcts_C" : self.mcts_C,
            "mcts_W" : self.mcts_W,
            "mcts_alpha_exploration" : self.mcts_alpha_exploration,
            "mcts_sim_step" : self.mcts_sim_step,
            "mcts_max_step_size" : self.mcts_max_step_size,
            "mcts_n_it" : self.mcts_n_it,
            "size_ratio" : self.size_ratio,
            "gait" : self.gait,
        }
        
        os.makedirs(saving_dir, exist_ok=True)
        file_path = os.path.join(saving_dir, ExperimentManager.CONFIG_FILE_NAME)

        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
    @staticmethod
    def load(experiment_path: str) -> 'ExperimentManager':
        file_path = os.path.join(experiment_path, ExperimentManager.CONFIG_FILE_NAME)

        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        manager = ExperimentManager(record_dir="", **config)
        manager.experiment_dir = os.path.split(file_path)[0]
        
        return manager