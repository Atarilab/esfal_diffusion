import os
import time
import numpy as np

from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from environment.stepping_stones import SteppingStonesEnv


### Data recorder
class ContactsDataRecorder(DataRecorderAbstract):
    FILE_NAME = "data.npz"
    STATE_NAME = "state"
    CONTACT_NAME = "contact_w"
    TARGET_NAME = "target_w"
    TARGET_ID_NAME = "target_id"
    STONES_NAME = "stones_w"
    GOAL_NAME = "goal_w"
    
    def __init__(self,
                 robot : MJQuadRobotWrapper,
                 stepping_stones_env : SteppingStonesEnv,
                 record_dir: str = "",
                 next_cnt_to_record : int = 2
                 ) -> None:
        super().__init__(record_dir)
        self.robot = robot
        self.stepping_stones = stepping_stones_env
        self.next_cnt_to_record = next_cnt_to_record
        
        self.update_record_dir(record_dir)

        # [x, y, z, qx, qy, qz, qw, qj, v, w, v_j]
        self.record_state = []
        # [c1, c2, c3, c4] in world frame
        self.record_feet_pos_w = []
        # [c1, c2, c3, c4] * next_cnt_to_record, stepping stones in world frame
        self.record_target_contact = []
        # [c1, c2, c3, c4] * next_cnt_to_record, stepping stones id
        self.record_target_contact_id = []
        
    def update_record_dir(self, record_dir:str):
        os.makedirs(record_dir, exist_ok=True)
        self.saving_file_path = os.path.join(record_dir, ContactsDataRecorder.FILE_NAME)

    def reset(self) -> None:
        self.record_state = []
        self.record_feet_pos_w = []
        self.record_target_contact = []
        self.record_target_contact_id = []

    def record(self, q: np.array, v: np.array, contact_plan_id : np.ndarray, i_jump : int) -> None:
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
        contact_locations_w = self.robot.get_foot_pos_world()
        self.record_feet_pos_w.append(contact_locations_w)
        
        if len(contact_plan_id) > 1:
            # Target contact locations w
            all_targets_contact_id = []
            for i_target in range(self.next_cnt_to_record):
                i_target = min(i_jump + i_target, len(contact_plan_id) - 1)
                all_targets_contact_id.extend(contact_plan_id[i_target])

            target_contact_w = self.stepping_stones.positions[all_targets_contact_id]
            self.record_target_contact.append(target_contact_w)
            
            # Target contact id
            self.record_target_contact_id.append(all_targets_contact_id)
            
        else:
            self.record_target_contact.append(np.zeros((self.next_cnt_to_record * 4, 3)))
            self.record_target_contact_id.append(np.zeros((self.next_cnt_to_record * 4)))
            

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
                    record_feet_contact = data[ContactsDataRecorder.CONTACT_NAME]
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
                ContactsDataRecorder.CONTACT_NAME : self.record_feet_pos_w,
                ContactsDataRecorder.TARGET_NAME : self.record_target_contact,
                ContactsDataRecorder.TARGET_ID_NAME : self.record_target_contact_id,
                ContactsDataRecorder.STONES_NAME : self.stepping_stones.positions,
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
        # Avoid too much repetitions in the dataset
        first_count, last_count = self.count_repeat()
        skip_first = max(0, first_count - 1)
        skip_last = max(0, last_count - 1)
        
        if lock:
            with lock:
                self._append_and_save(skip_first, skip_last)
        else:
            self._append_and_save(skip_first, skip_last)

        self.reset()