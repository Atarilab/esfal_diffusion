import os
import numpy as np
import tyro

from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.bicon_mpc import BiConMPC
from environment.simulator import SteppingStonesSimulator
from environment.stepping_stones import SteppingStonesEnv
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from utils.rendering import desired_contact_locations_callback
from utils.config import Go2Config
from tree_search.data_recorder import ContactsDataRecorder
from tree_search.mcts_stepping_stones import MCTSSteppingStonesKin


### Example code for the whole pipeline
# - Create a randomized stepping stone environment
# - Create a MuJoCo / Pinocchio robot wrapper
# - Create a MuJoCo simulator
# - Perform search with MCTS
# - Visualize contact plan if found

def main(
    stepping_stones_height : float = 0.1,
    randomize_height_ratio : float = 0.2,
    randomize_pos_ratio : float = 0.8,
    size_ratio : float = 0.55,
    N_to_remove : int = 7,
    gait : str = "jump", # or trot
    max_step_size : float = 0.23, 
    data_path : str = "../data/test",
    ):
    
    # Stepping stones environment
    stones_env = SteppingStonesEnv(
        spacing=(0.19, 0.13),
        height=stepping_stones_height,
        randomize_height_ratio=randomize_height_ratio,
        randomize_pos_ratio=randomize_pos_ratio,
        size_ratio=(size_ratio, size_ratio),
        N_to_remove=N_to_remove,
        )

    # Robot wrapper
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
    )
    
    # BiconMP controller
    controller = BiConMPC(robot.pin,
                          replanning_time=0.05,
                          sim_opt_lag=False,
                          height_offset=stepping_stones_height,
                          )
    # Set gait
    if gait == "trot":
        controller.set_gait_params(trot)
    else:
        controller.set_gait_params(jump)
        
    # Data recorder
    if data_path != "":
        os.makedirs(data_path, exist_ok=True)
        data_recorder = ContactsDataRecorder(robot.mj, stones_env, data_path)
    else:
        data_recorder = None

    # Simulator
    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot.mj,
        controller=controller,
        data_recorder=data_recorder
        )
    
    # MCTS
    mcts = MCTSSteppingStonesKin(
        sim,
        simulation_steps=1,
        C=1.0e-1,
        W=1.,
        alpha_exploration=0.,
        max_step_size=max_step_size,
        max_solution_search=1
    )
    
    # Set random goal
    mcts.sim.set_start_and_goal()
    
    # Perform the search
    mcts.search(20000)

    # Visualize contact plan found
    if len(mcts.all_solutions) > 0:
        contact_plan = mcts.all_solutions[0]
        contact_plan_callback = (lambda viewer, step, q, v, data :
            desired_contact_locations_callback(viewer, step, q, v, data, controller)
            )
        
        success = sim.run_contact_plan(
            contact_plan,
            use_viewer=True,
            visual_callback_fn=contact_plan_callback,
            real_time=False,
            verbose=True)
        
        if success > 0: print("Success")
        else: print("Failure")
        
    else:
        print("No solution found")
        
if __name__ == "__main__":
    args = tyro.cli(main)