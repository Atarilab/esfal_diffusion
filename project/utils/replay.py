import os

from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from environment.stepping_stones import SteppingStonesEnv
from environment.simulator import SteppingStonesSimulator
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.raibert import MPC_RaiberContactPlanner
from mpc_controller.learned import MPC_LearnedContactPlanner
from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from tree_search.experiment_manager import ExperimentManager
from utils.rendering import desired_contact_locations_callback
from utils.eval import PerformanceGatherer
from utils.config import Go2Config

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
            model_path : str = "",
            i_env : int = 0,
            i_goal : int = 0,
            i_cp : int = 0,
            use_viewer : bool = True,
            record_video : bool = True,
            video_subdir : str = "",
            ):
        
        # Get contact plan (if found)
        env_dir, goal_dir = self.mcts_perfs.get_env_goal_paths(i_env, i_goal)
        run_data = self.mcts_perfs.get_data(i_env, i_goal)
        all_contact_plans = run_data["contact_plan"]
        
        if len(all_contact_plans) == 0:
            print("No contact plan found in", goal_dir)
            return None
        
        contact_plan = all_contact_plans[min(i_cp, len(all_contact_plans))]

        # Setup environment
        stones_env = SteppingStonesEnv.load(env_dir)
        
        cfg = Go2Config
        robot_paths = *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        robot = MJPinQuadRobotWrapper(
            *robot_paths,
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio
            )
        
        # Setup controller
        if model_path != "":
            controller_ = MPC_LearnedContactPlanner(robot.pin, stones_env, model_path, height_offset=stones_env.height)
        elif controller == "biconmp":
            controller_ = BiConMPC(robot.pin, height_offset=stones_env.height)
        elif controller == "raibert":
            controller_ = MPC_RaiberContactPlanner(robot.pin, stones_env, v_max=0.35, height_offset=stones_env.height)
        else:
            print(f"{controller} is not a valid controller name. Using default BiConMP")
            controller_ = BiConMPC(robot.pin, height_offset=stones_env.height)

        gait = trot if self.gait_str == "trot" else jump
        controller_.set_gait_params(gait)
        
        # Setup simulator
        sim = SteppingStonesSimulator(
            stepping_stones_env=stones_env,
            robot=robot.mj,
            controller=controller_,
            )
               
        # Record video dir
        if video_subdir == "":
            exp_name = f"{self.experiment_name}_{i_env}_{i_goal}_{controller}"
            video_save_dir = os.path.join(ExperimentReplay.VIDEO_RECORD_DIR, exp_name)
        else:
            video_save_dir = os.path.join(ExperimentReplay.VIDEO_RECORD_DIR, video_subdir)
            
        os.makedirs(video_save_dir, exist_ok=True)
        
        # Run simulation
        contact_plan_callback = (lambda viewer, step, q, v, data :
            desired_contact_locations_callback(viewer, step, q, v, data, controller_)
            )
        if controller =="biconmp" and model_path == "":     
            success = sim.run_contact_plan(contact_plan,
                                        use_viewer=use_viewer,
                                        visual_callback_fn=contact_plan_callback,
                                        record_video = record_video,
                                        video_save_dir = video_save_dir
                                        )
        else:
            success = sim.reach_goal(contact_plan[-1],
                                    use_viewer=use_viewer,
                                    visual_callback_fn=contact_plan_callback,
                                    record_video = record_video,
                                    video_save_dir = video_save_dir
                                    )
        
        if success > 0: print("Success")
        else: print("Failure")