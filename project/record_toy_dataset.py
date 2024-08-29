
import multiprocessing
import tqdm
import time
import numpy as np
import gc
import tyro

from environment.stepping_stones import SteppingStonesEnv
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from tree_search.data_recorder import ContactsDataRecorder
from environment.simulator import SteppingStonesSimulator, Simulator
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from utils.config import Go2Config

def record_one_environment(path, i_env, gait, robot_paths):
    # path = args[0]
    # i_env = args[1]
    # robot_paths = args[2]
    
    seed = int(time.time() * 33333) % 11111 * (33 + i_env**2)
    np.random.seed(seed)
    
    # Regular grid environment
    stepping_stones_height = 0.1
    randomize_height_ratio = 0.
    randomize_pos_ratio = 0.
    size_ratio = 0.65
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
        *robot_paths,
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
    )

    controller = BiConMPC(robot.pin, height_offset=stepping_stones_height)
    
    if gait == "trot":
        controller.set_gait_params(trot)
    else:
        controller.set_gait_params(jump)
    
    data_recorder = ContactsDataRecorder(robot.mj, stones_env, f"{path}/env_{i_env}/goal_0/")
    
    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot.mj,
        controller=controller,
        data_recorder=data_recorder,
        )
    
    start_indices = [50, 32, 48, 30]
    contact_plan_id = np.array([start_indices, start_indices, start_indices, [51, 33, 49, 31], [52, 34, 50, 32]])
        
    sim.run_contact_plan(contact_plan_id, use_viewer=False, verbose=False, randomize=True)

    # Clean up to release memory
    del sim
    del data_recorder
    del controller
    del robot
    del stones_env
    gc.collect()
    
def multiprocess(n_cores : int, n_runs : int, gait : str, record_path : str):
    processes = []
    cfg = Go2Config
    robot_paths = *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),

    for i in tqdm.trange(n_runs):
        p = multiprocessing.Process(target=record_one_environment, args=(record_path, i, gait, robot_paths))
        processes.append(p)
        p.start()

        # Limit the number of concurrent processes
        if len(processes) >= n_cores:
            for p in processes:
                p.join()
            processes = []

    # Join any remaining processes
    for p in processes:
        p.join()
    
def main(n_cores: int = 20,
         n_runs: int = 100,
         gait: str = "jump",
         data_dir_name: str = "forward"):
    """
    Record simple dataset with one goal forward.
    """
    path = f"../data/{gait}_{data_dir_name}/train"
    multiprocess(n_cores, n_runs, gait, path)
    
    n_runs = int(n_runs * 0.2)
    path = f"../data/{gait}_{data_dir_name}/test"
    multiprocess(n_cores, n_runs, gait, path)

if __name__ == "__main__":
    args = tyro.cli(main)