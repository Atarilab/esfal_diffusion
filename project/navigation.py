import tyro

from environment.stepping_stones import SteppingStonesEnv
from environment.simulator import SteppingStonesSimulator, NavigationSteppingStonesSimulator
from mpc_controller.bicon_mpc import BiConMPC
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mpc_controller.raibert import MPC_RaiberContactPlanner
from mpc_controller.learned import MPC_LearnedContactPlanner
from utils.rendering import desired_contact_locations_callback
from utils.config import Go2Config
from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump


def main(
    planner : str = "raibert",
    model_path : str = "",
    gait : str = "jump",
    stepping_stones_height :float = 0.1,
    randomize_height_ratio :float = 0.,
    randomize_pos_ratio :float = 0.,
    size_ratio :float = 0.65,
    project_closest : bool = True,
    diffusion_steps : int = -1,
    ):

    stones_env = SteppingStonesEnv(
        spacing=(0.19, 0.13),
        height=stepping_stones_height,
        randomize_height_ratio=randomize_height_ratio,
        randomize_pos_ratio=randomize_pos_ratio,
        size_ratio=(size_ratio, size_ratio),
        N_to_remove=0
        )

    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )

    if model_path != "":
        print("--- Learned policy from", model_path)
        controller = MPC_LearnedContactPlanner(
            robot.pin,
            stones_env,
            model_path,
            project = project_closest,
            diffusion_steps = diffusion_steps,
            height_offset=stepping_stones_height)

    elif planner == "raibert":
        print("--- Raibert-Based contact planner")
        controller = MPC_RaiberContactPlanner(robot.pin, stones_env, v_max=0.33, height_offset=stepping_stones_height)

    else:
        assert False, "Specify a valid planner."
    
    gait_controller = trot if gait == "trot" else jump
    controller.set_gait_params(gait_controller)

    sim = NavigationSteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot.mj,
        controller=controller,
        )

    contact_plan_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller)
        )
    
    success = sim.run(real_time=False, visual_callback_fn=contact_plan_callback)
    print("Success:", success)
    
if __name__ == "__main__":
    args = tyro.cli(main)