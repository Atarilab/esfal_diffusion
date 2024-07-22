import ast  # For literal_eval
import csv
import glob
import os
import pickle
import time
import pybullet
import multiprocessing
import tqdm

from environment.simulator import SteppingStonesSimulator
from environment.stepping_stones import SteppingStonesEnv
from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.motions.cyclic.solo12_trot import trot
from mpc_controller.motions.cyclic.solo12_jump import jump
from tree_search.mcts_stepping_stones import MCTSSteppingStonesKin
from tree_search.data_recorder import JumpDataRecorder
from tree_search.experiment_manager import ExperimentManager
from mpc_controller.raibert import MPC_RaiberContactPlanner
from mpc_controller.learned import MPC_LearnedContactPlanner

class PerformanceGatherer:
    def __init__(self, experiment_dir : str):
        self.experiment_dir = experiment_dir
        self.file_name = MCTSSteppingStonesKin.MCTS_PERF_FILE_NAME
        
        self.data = {}
        self.gather_performance_data()
        
    @staticmethod
    def get_all_run_dir(experiment_dir : str) -> list:
        all_run_dir = []
        for env_name in os.listdir(experiment_dir):
            env_dir = os.path.join(experiment_dir, env_name)
            if not os.path.isdir(env_dir):
                continue
            
            for goal_name in os.listdir(env_dir):
                goal_dir = os.path.join(env_dir, goal_name)
                if not os.path.isdir(goal_dir):
                    continue
                
                # If there is perf file
                if len(glob.glob(f"{goal_dir}/*.pkl")) > 0:
                    all_run_dir.append(goal_dir)
                    
        return all_run_dir
                
    def gather_performance_data(self):
        self.data = {}

        for goal_dir in self.get_all_run_dir(self.experiment_dir):
                file_path = os.path.join(goal_dir, self.file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        performance_data = pickle.load(f)
                        for key, value in performance_data.items():
                            if key not in self.data:
                                self.data[key] = []
                            self.data[key].extend([value])
                            
        self.data["success"] = [int(len(i) > 0) for i in self.data["n_nmpc_first"]]
    
    def get_env_goal_paths(self, i_env : int, i_goal : int) -> str:
        env_name = f"env_{i_env}"
        env_dir = os.path.join(self.experiment_dir, env_name)
        if not os.path.exists(env_dir): env_dir = ""
        
        goal_name = f"goal_{i_goal}"
        goal_dir = os.path.join(env_dir, goal_name)
        if not os.path.exists(goal_dir): goal_dir = ""
        
        return env_dir, goal_dir
        
    def get_data(self, i_env, i_goal) -> dict:
        data_env_goal = {}
        
        _, goal_dir = self.get_env_goal_paths(i_env, i_goal)
    
        file_path = os.path.join(goal_dir, self.file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                data_env_goal = pickle.load(f)

        return data_env_goal

class RerunExperiments:
    VALID_CONTACT_PLANNERS = {
        "raibert" : MPC_RaiberContactPlanner,
        "learned" : MPC_LearnedContactPlanner
        }
    def __init__(self,
                 experiment_dir : str,
                 contact_planner : str = "raibert",
                 model_path : str = "") -> None:
        
        self.experiment_dir = experiment_dir
        
        date_time_str = time.strftime('%d%m-%H%M%S')
        self.rerun_experiment_dir = os.path.join(experiment_dir, f"_rerun_{contact_planner}_{date_time_str}")
        
        assert contact_planner in RerunExperiments.VALID_CONTACT_PLANNERS.keys(), f"{contact_planner} is not a valid name"
        self.contact_planner = contact_planner
        self.model_path = model_path
        
        self.manager = ExperimentManager.load(self.experiment_dir)
        self.mcts_perf = PerformanceGatherer(self.experiment_dir)
    
    def init_experiment_rerun(self, i_env : int, i_goal : int) -> SteppingStonesSimulator:
        """
        Setup an experiment rerun (same env, same goal).
        """
        env_dir = os.path.join(self.experiment_dir, f"env_{i_env}")
        goal_dir = os.path.join(env_dir, f"goal_{i_goal}")
        
        if not os.path.exists(goal_dir):
            return None
        
        rerun_env_dir = os.path.join(self.rerun_experiment_dir, f"env_{i_env}")
        rerun_goal_dir = os.path.join(rerun_env_dir, f"goal_{i_goal}")
        
        os.makedirs(rerun_env_dir, exist_ok=True)
        os.makedirs(rerun_goal_dir,exist_ok=True)
        
        stepping_stones_env = SteppingStonesEnv.load(env_dir)
        
        robot = SoloRobotWrapper(server = pybullet.DIRECT)

        controller = RerunExperiments.VALID_CONTACT_PLANNERS[self.contact_planner](
            robot = robot,
            stepping_stones_env = stepping_stones_env,
            model_path = self.model_path,
            height_offset = stepping_stones_env.height
            )
        gait = jump if self.manager.gait == "jump" else trot
        controller.set_gait_params(gait)
                
        data_recorder = JumpDataRecorder(
            robot,
            stepping_stones_env,
            record_dir=rerun_goal_dir,
            )
        
        sim = SteppingStonesSimulator(
            stepping_stones_env=stepping_stones_env,
            robot=robot,
            controller=controller,
            data_recorder=data_recorder
            )
        
        return sim
    
    def rerun_single_experiment(self, args):
        i_env = args[0]
        i_goal = args[1]

        sim = self.init_experiment_rerun(i_env, i_goal)
        if sim is None:
            return -1
        
        mcts_data_run = self.mcts_perf.get_data(i_env, i_goal) 
        if MCTSSteppingStonesKin.GOAL in mcts_data_run.keys():
            goal_id = mcts_data_run[MCTSSteppingStonesKin.GOAL]
            success = sim.reach_goal(goal_id)
        else:
            return -1
        
        return success
    
    def start(self, n_cores : int = 10):
        """
        Launch experiment on different cores.
        """
        all_args = []
        i_env = 0
        for env_name in os.listdir(self.experiment_dir):
            env_dir = os.path.join(self.experiment_dir, env_name)
            if not os.path.isdir(env_dir):
                continue
            
            i_goal = 0
            for goal_name in os.listdir(env_dir):
                goal_dir = os.path.join(env_dir, goal_name)
                if not os.path.isdir(goal_dir):
                    continue
                
                mcts_perf_path = os.path.join(goal_dir, MCTSSteppingStonesKin.MCTS_PERF_FILE_NAME)
                if os.path.exists(mcts_perf_path):
                    args = (i_env, i_goal)
                    all_args.append(args)
                i_goal += 1
                    
            i_env += 1

        n_runs = len(all_args)
        with multiprocessing.Pool(n_cores) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(self.rerun_single_experiment, all_args), total=n_runs):
                pass

def load_results(file):
    """
    Load results from .txt file formatted as so:
    <solution_name>, <success>, <contact_plan_model>, <contact_plan_mcts>

    Returns a dict with each id as a key and remaining data in dict 
    {solution_name:, success: , cp_model:, cp_mcts:}
    """
    result_data = {}

    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        for id, row in enumerate(csv_reader):
            solution_name = row[0] # Unique key
            contact_plan_mcts = ast.literal_eval(row[1])
            success = row[-1]
            contact_plan_model = ast.literal_eval(row[2]) if len(row) >= 4 else None
            
            result_data[id] = {
                "name" : solution_name,
                "success" : int(success),
                "cp_model" : contact_plan_model,
                "cp_mcts" : contact_plan_mcts
            }
    return result_data

def count_success(res_data):
    count = 0
    total = 0
    for val in iter(res_data.values()):
        count += val["success"]
        total += 1

    print(f"{count} success on {total} attemps ({count / total * 100:.1f}%)")
    return count, total

def compare_contact_plans(res_model):
    same_plan = 0
    shorter_plan = 0
    mean_length_model = 0
    mean_length_mcts = 0
    total = 0

    def is_subplan(cpA, cpB):
        """
        True is contact plan cpA is a subplan of cpB.
        """
        if len(cpB) < len(cpA):
            return False
        
        for i in range(len(cpB) - len(cpA)):
            if cpB[i:i+len(cpA)] == cpA:
                return True
            
        return False
    
    def clear_first_last(cpA):
        """
        Remove first and last position of the plan.
        """
        cpA = [cp for cp in cpA if (cp != cpA[0] and cp != cpA[-1])]
        return cpA
               
    for val in iter(res_model.values()):
        if val["success"]:
            cp_model = val["cp_model"]
            cp_mcts = val["cp_mcts"]

            if cp_model == cp_mcts or \
              clear_first_last(cp_model) == clear_first_last(cp_mcts) or \
              is_subplan(cp_model, cp_mcts) or \
              is_subplan(cp_mcts, cp_model):
                same_plan += 1

            if len(cp_model) < len(cp_mcts):
                shorter_plan += 1


            mean_length_model += len(cp_model)
            mean_length_mcts += len(cp_mcts)

            total += 1

    mean_length_model /= total
    mean_length_mcts /= total

    print(f"{same_plan} similar plans ({same_plan / total * 100:.1f}%)")
    print(f"{shorter_plan} shorter plans ({shorter_plan / total * 100:.1f}%)")
    print(f"Mean contact plan lenght, model: {mean_length_model:.1f}, mcts:{mean_length_mcts:.1f}")

    return same_plan, shorter_plan, total

def compare_with_mcts(path_res_model):
    res_model = load_results(path_res_model)

    print("--- Model")
    count_model, total_count = count_success(res_model)

    print("--- Contact plan")
    same_cp, shorter_cp, total = compare_contact_plans(res_model)