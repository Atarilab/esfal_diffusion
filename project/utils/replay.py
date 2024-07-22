import os
import pybullet

from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.raibert import MPC_RaiberContactPlanner
from environment.stepping_stones import SteppingStonesEnv
from environment.simulator import SteppingStonesSimulator
from mpc_controller.motions.cyclic.solo12_trot import trot
from mpc_controller.motions.cyclic.solo12_jump import jump
from utils.rendering import desired_contact_locations_callback
from tree_search.experiment_manager import ExperimentManager
from utils.eval import PerformanceGatherer

class ExperimentReplay():
    VIDEO_RECORD_DIR = "/home/atari_ws/project/figures/videos"
    VALID_CONTROLLER = ["", "biconmp", "raibert"]
    
    def __init__(self, experiment_dir : str) -> None:
        self.experiment_dir = experiment_dir
        self.mcts_perfs = PerformanceGatherer(self.experiment_dir)
        self.experiment_name = os.path.split(self.experiment_dir)[-1]
        exp_manager = ExperimentManager.load(self.experiment_dir)
        self.gait_str = exp_manager.gait
        
    def run(self,
            controller : str = "biconmp",
            i_env : int = 0,
            i_goal : int = 0,
            i_cp : int = 0,
            use_viewer : bool = True,
            record_video : bool = True,
            video_subdir : str = "",
            ):
        
        env_dir, _ = self.mcts_perfs.get_env_goal_paths(i_env, i_goal)
        run_data = self.mcts_perfs.get_data(i_env, i_goal)
        all_contact_plans = run_data["contact_plan"]
        
        if len(all_contact_plans) == 0:
            return None
        
        contact_plan = all_contact_plans[min(i_cp, len(all_contact_plans))]

        stones_env = SteppingStonesEnv.load(env_dir)
        
        robot = SoloRobotWrapper(server = pybullet.GUI if use_viewer else pybullet.DIRECT)
        
        if controller == "biconmp":
            controller = BiConMPC(robot, height_offset=stones_env.height)
        elif controller == "raibert":
            controller = MPC_RaiberContactPlanner(robot, stones_env, v_max=0.22, height_offset=stones_env.height)
        else:
            print(f"{controller} is not a valid controller name. Using default BiConMP")
            controller = BiConMPC(robot, height_offset=stones_env.height)

        gait = trot if self.gait_str == "trot" else jump
        controller.set_gait_params(gait)
                
        sim = SteppingStonesSimulator(
            stepping_stones_env=stones_env,
            robot=robot,
            controller=controller,
            )
        
        contact_plan_callback = lambda env, sim_step, q, v : desired_contact_locations_callback(env, sim_step, q, v, controller)
        
        if video_subdir == "":
            exp_name = f"{self.experiment_name}_{i_env}_{i_goal}"
            video_save_dir = os.path.join(ExperimentReplay.VIDEO_RECORD_DIR, exp_name)
        else:
            video_save_dir = os.path.join(ExperimentReplay.VIDEO_RECORD_DIR, video_subdir)
            
        os.makedirs(video_save_dir, exist_ok=True)
            
        
        success = sim.run_contact_plan(contact_plan,
                                       use_viewer=use_viewer,
                                       visual_callback_fn=contact_plan_callback,
                                       record_video = record_video,
                                       video_save_dir = video_save_dir
                                       )
                
        print(success)