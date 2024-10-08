import tyro

from environment.stepping_stones import SteppingStonesEnv
from environment.simulator import SteppingStonesSimulator
from mpc_controller.bicon_mpc import BiConMPC
from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.raibert import MPC_RaiberContactPlanner
from mpc_controller.learned import MPC_LearnedContactPlanner
from utils.rendering import desired_contact_locations_callback
from tree_search.data_recorder import ContactsDataRecorder
from mpc_controller.motions.cyclic.solo12_trot import trot
from mpc_controller.motions.cyclic.solo12_jump import jump


def main(
    contact_planner : str = "raibert",
    model_path : str = "",
    gait : str = "jump",
    stepping_stones_height :float = 0.1,
    randomize_height_ratio :float = 0.25,
    randomize_pos_ratio :float = 0.4,
    size_ratio :float = 0.65,
    project_closest : bool = True,
    diffusion_steps : int = -1,
    ):

    stones_env = SteppingStonesEnv(
        height=stepping_stones_height,
        randomize_height_ratio=randomize_height_ratio,
        randomize_pos_ratio=randomize_pos_ratio,
        size_ratio=(size_ratio, size_ratio),
        N_to_remove=0
        )

    robot = SoloRobotWrapper()

    if model_path != "":
        controller = MPC_LearnedContactPlanner(
            robot,
            stones_env,
            model_path,
            project = project_closest,
            diffusion_steps = diffusion_steps,
            height_offset=stepping_stones_height)

    elif contact_planner == "raibert":
        controller = MPC_RaiberContactPlanner(robot, stones_env, v_max=0.33, height_offset=stepping_stones_height)

    else:
        controller =  BiConMPC(robot, height_offset=stepping_stones_height)
    
    gait_controller = jump if gait == "jump" else trot
    controller.set_gait_params(gait_controller)
    #data_recorder = JumpDataRecorder(robot, stones_env, "test")

    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot,
        controller=controller,
        )

    goal_indices = [51, 33, 49, 31]
    goal_indices = [52, 34, 50, 32]
    goal_indices = []
    contact_plan_callback = lambda env, sim_step, q, v : desired_contact_locations_callback(env, sim_step, q, v, controller)
    success = sim.reach_goal(goal_indices, use_viewer=True, visual_callback_fn=contact_plan_callback, verbose=True)
    print("Success:", success)
    
if __name__ == "__main__":
    args = tyro.cli(main)