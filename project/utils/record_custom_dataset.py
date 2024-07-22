import pybullet 
import multiprocessing
import tqdm
import numpy as np
import time
import tyro
import os

from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.motions.cyclic.solo12_trot import trot
from mpc_controller.motions.cyclic.solo12_jump import jump
from tree_search.data_recorder import JumpDataRecorder
from environment.simulator import SteppingStonesSimulator
from environment.stepping_stones import SteppingStonesEnv


def record_one_environement(args):
    path = args[0]
    i_env = args[1]
    
    seed = int(time.time() * 337) % (i_env * 77 + 1) + i_env
    np.random.seed(seed)

    stepping_stones_height = 0.1
    stones_env = SteppingStonesEnv(
        height=stepping_stones_height,
        randomize_height_ratio=0.3,
        randomize_pos_ratio=0.,
        size_ratio=(0.7, 0.7),
        N_to_remove=0
        )
    
    robot = SoloRobotWrapper(server=pybullet.DIRECT)

    controller = BiConMPC(robot, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)
    
    data_recorder = JumpDataRecorder(robot, stones_env, f"{path}/env_{i_env}/goal_0/")
    
    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot,
        controller=controller,
        data_recorder=data_recorder,
        )
    
    start_indices = [50, 32, 48, 30]
    # contact_plan_id = np.array([start_indices] * 20)
    contact_plan_id = np.array([[50, 32, 48, 30], [50, 32, 48, 30], [51, 33, 49, 31], [52, 34, 50, 32]])
        
    # contact_plan_callback = lambda env, sim_step, q, v : desired_contact_locations_callback(env, sim_step, q, v, controller)
    success = sim.run_contact_plan(contact_plan_id, use_viewer=False, verbose=False, randomize=True)

def main(
    record_dir : str,
    n_cores : int = 20,
    n_runs : int = 100
):
    """
    Record simple dataset with one goal.
    """
    # Record train
    path = os.path.join(record_dir, "train")
    os.makedirs(path, exist_ok=True)
    with multiprocessing.Pool(n_cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(record_one_environement, [(path, i) for i in range(n_runs)]), total=n_runs):
            pass
    
    # Record test
    n_runs = n_runs // 5
    path = os.path.join(record_dir, "test")
    os.makedirs(path, exist_ok=True)
    with multiprocessing.Pool(n_cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(record_one_environement, [(path, i) for i in range(n_runs)]), total=n_runs):
            pass
        
if __name__ == "__main__":
    args = tyro.cli(main)