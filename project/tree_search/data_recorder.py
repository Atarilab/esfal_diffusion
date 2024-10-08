import os
import time
import numpy as np

from py_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from environment.stepping_stones import SteppingStonesEnv


### Data recorder
class ContactsDataRecorder(DataRecorderAbstract):
    FILE_NAME = "data.npz"
    STATE_NAME = "state"
    FEET_POS_NAME = "feet_pos_w"
    TARGET_NAME = "target_w"
    TARGET_ID_NAME = "target_id"
    CONTACT_NAME = "contact_w"
    GOAL_NAME = "goal_w"
    
    def __init__(self,
                 robot : SoloRobotWrapper,
                 stepping_stones_env : SteppingStonesEnv,
                 record_dir: str = "",
                 next_target : int = 2
                 ) -> None:
        super().__init__(record_dir)
        self.robot = robot
        self.stepping_stones = stepping_stones_env
        self.next_target = next_target
        self.i_jump = 0
        
        self.update_record_dir(record_dir)

        # [x, y, z, qx, qy, qz, qw, qj, v, w, v_j]
        self.record_state = []
        # [c1, c2, c3, c4] in world frame
        self.record_feet_pos_w = []
        # [c1, c2, c3, c4] * next_target, stepping stones in world frame
        self.record_target_contact = []
        # [c1, c2, c3, c4] * next_target, stepping stones id
        self.record_target_contact_id = []
        
    def update_record_dir(self, record_dir:str):
        os.makedirs(record_dir, exist_ok=True)
        self.saving_file_path = os.path.join(record_dir, ContactsDataRecorder.FILE_NAME)

    def reset(self) -> None:
        self.i_jump = 0
        self.record_state = []
        self.record_feet_pos_w = []
        self.record_target_contact = []
        self.record_target_contact_id = []

    def record(self, q: np.array, v: np.array, contact_plan_id : np.ndarray) -> None:
        """ 
        Record state, current contact, target contact locations / id.
        All expressed in world frame.
        """
        # State
        current_state = np.concatenate((q, v), axis=0)
        self.record_state.append(current_state)
        # Goal
        self.goal_locations_w = self.stepping_stones.positions[contact_plan_id[-1]]

        # Current contacts
        contact_locations_w = self.robot.get_foot_locations_world()
        self.record_feet_pos_w.append(contact_locations_w)

        if len(contact_plan_id) > 1:
            # Target contact locations w
            all_targets_contact_id = []
            for i_target in range(self.next_target):
                i_target = min(self.i_jump + i_target, len(contact_plan_id) - 1)
                all_targets_contact_id.extend(contact_plan_id[i_target])

            target_contact_w = self.stepping_stones.positions[all_targets_contact_id]
            self.record_target_contact.append(target_contact_w)
            
            # Target contact id
            self.record_target_contact_id.append(all_targets_contact_id)
        else:
            self.record_target_contact.append(np.zeros((self.next_target * 4, 3)))
            self.record_target_contact_id.append(np.zeros((self.next_target * 4)))
            
        self.i_jump += 1

    def _append_and_save(self, skip_first, skip_last):
        """ 
        Append new data to existing file and save file.
        """
        N = len(self.record_state)
        if N - skip_first - skip_last > 0:

            # skip first / last              
            self.record_state = self.record_state[skip_first:N-skip_last]
            self.record_feet_pos_w = self.record_feet_pos_w[skip_first:N-skip_last]
            self.record_target_contact = self.record_target_contact[skip_first:N-skip_last]
            self.record_target_contact_id = self.record_target_contact_id[skip_first:N-skip_last]
            
            # Load data and append if exists
            if os.path.exists(self.saving_file_path):
                
                data = np.load(self.saving_file_path)
                if data.keys():
                    record_state = data[ContactsDataRecorder.STATE_NAME]
                    record_feet_contact = data[ContactsDataRecorder.FEET_POS_NAME]
                    record_target_contact = data[ContactsDataRecorder.TARGET_NAME]
                    record_target_contact_id = data[ContactsDataRecorder.TARGET_ID_NAME]
                    
                    # Concatenate
                    self.record_state = np.concatenate((record_state, self.record_state), axis = 0)
                    self.record_feet_pos_w = np.concatenate((record_feet_contact, self.record_feet_pos_w), axis = 0)
                    self.record_target_contact = np.concatenate((record_target_contact, self.record_target_contact), axis = 0)
                    self.record_target_contact_id = np.concatenate((record_target_contact_id, self.record_target_contact_id), axis = 0)
            
            # Save with new data / save stones position /!\ overrides it
            d = {
                ContactsDataRecorder.STATE_NAME : self.record_state,
                ContactsDataRecorder.GOAL_NAME : self.goal_locations_w,
                ContactsDataRecorder.FEET_POS_NAME : self.record_feet_pos_w,
                ContactsDataRecorder.TARGET_NAME : self.record_target_contact,
                ContactsDataRecorder.TARGET_ID_NAME : self.record_target_contact_id,
                ContactsDataRecorder.CONTACT_NAME : self.stepping_stones.positions,
            }
            np.savez(self.saving_file_path, **d)
            

    def count_repeat(self):
        """
        Count consecutive repetitions of the first and last values in the recorded data.
        """
        if len(self.record_target_contact_id) == 0:
            return 0, 0

        first_value = self.record_target_contact_id[0]
        last_value = self.record_target_contact_id[-1]

        first_count = 0
        last_count = 0

        # Count consecutive repetitions of the first value
        for value in self.record_target_contact_id:
            if np.array_equal(value, first_value):
                first_count += 1
            else:
                break

        # Count consecutive repetitions of the last value
        if not np.array_equal(first_value, last_value):
            for value in reversed(self.record_target_contact_id):
                if np.array_equal(value, last_value):
                    last_count += 1
                else:
                    break

        return first_count, last_count

    
    def save(self, lock = None) -> None:
        first_count, last_count = self.count_repeat()

        skip_first = max(0, first_count - 1)
        skip_last = max(0, last_count - 1)
        
        if lock:
            with lock:
                self._append_and_save(skip_first, skip_last)
        else:
            self._append_and_save(skip_first, skip_last)

        self.reset()
        

import pybullet 
import multiprocessing
import tqdm

from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.motions.cyclic.solo12_trot import trot
from mpc_controller.motions.cyclic.solo12_jump import jump
from tree_search.data_recorder import ContactsDataRecorder
from environment.simulator import SteppingStonesSimulator



def record_one_environement(args):
    path = args[0]
    i_env = args[1]
    
    seed = int(time.time() * 337) % 33 + i_env
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
    
    data_recorder = ContactsDataRecorder(robot, stones_env, f"{path}/env_{i_env}/goal_0/")
    
    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot,
        controller=controller,
        data_recorder=data_recorder,
        )
    
    start_indices = [51, 33, 49, 31]
    contact_plan_id = np.array([start_indices] * 20)
    contact_plan_id = np.array([[51, 33, 49, 31], [51, 33, 49, 31], [51, 33, 49, 31], [52, 34, 50, 32], [53, 35, 51, 33]])
        
    # contact_plan_callback = lambda env, sim_step, q, v : desired_contact_locations_callback(env, sim_step, q, v, controller)
    sim.run_contact_plan(contact_plan_id, use_viewer=False, verbose=False, randomize=True)


def main():
    """
    Record simple dataset with one goal.
    """

    n_cores = 20
    n_runs = 100
    path = "../data/one_goal/train"
    with multiprocessing.Pool(n_cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(record_one_environement, [(path, i) for i in range(n_runs)]), total=n_runs):
            pass
        
    n_cores = 20
    n_runs = 20
    path = "../data/one_goal/test"
    with multiprocessing.Pool(n_cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(record_one_environement, [(path, i) for i in range(n_runs)]), total=n_runs):
            pass
        
if __name__ == "__main__":
    main()