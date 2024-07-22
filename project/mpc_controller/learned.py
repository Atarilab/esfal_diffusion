import copy
from typing import Dict, Tuple
import numpy as np
import pinocchio as  pin 
import torch
import os
import pickle

from environment.stepping_stones import SteppingStonesEnv
from mpc_controller.bicon_mpc import BiConMPC
from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from learning.utils.utils import get_model, get_config

NORMALIZATION_STATS = "normalization_stats.pkl"
STATE_NAME = "state"
VELOCITY_NAME = "velocity"
GOAL_NAME = "goal_w"
CONTACT_NAME = "contact_w"
TARGET_NAME = "target_w"
TARGET_ID_NAME = "target_id"
STONES_NAME = "stones_w"

class MPC_LearnedContactPlanner(BiConMPC):
    DEFAULT_PROJECT = True
    DEFAULT_DIFFUSION_STEPS = -1
    KEYS_ORDERED = [
        GOAL_NAME,
        CONTACT_NAME,
        VELOCITY_NAME,
        STATE_NAME,
        STONES_NAME,
    ]
    def __init__(self,
                 robot: SoloRobotWrapper,
                 stepping_stones_env: SteppingStonesEnv,
                 model_path : str = "",
                 **kwargs) -> None:
        
        # Optional arguments
        optionals_args = {
            "project" : MPC_LearnedContactPlanner.DEFAULT_PROJECT,
            "diffusion_steps" : MPC_LearnedContactPlanner.DEFAULT_DIFFUSION_STEPS,
        }
        optionals_args.update(kwargs)
        for (k, v) in optionals_args.items(): setattr(self, k, v)
        
        super().__init__(robot, **kwargs)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.goal_locations = None
        self.stones_env = stepping_stones_env
        self.i_cnt_replan = 0
        self.model_path = model_path
        self.model = get_model(state_path=model_path).to(self.device)
        
        self.data_stats = {}
        self.load_data_stats()
        
    def load_data_stats(self):
        cfg = get_config(state_path=self.model_path)
        cfg_dict = cfg.get_cfg_as_dict()
        normalized = cfg_dict.get("normalized", False)
        
        if normalized:
            data_dir = cfg_dict.get("data_dir", "")
            dataset = cfg_dict.get("dataset", "")
            dataset_dir = os.path.join(data_dir, dataset)
            data_stats_file = os.path.join(dataset_dir,  NORMALIZATION_STATS)
            
            if os.path.exists(data_stats_file):
                self.data_stats = self.load_data_stats_file(data_stats_file)
                self.get_mean_std_vectors()
            
    def load_data_stats_file(self, file_path : str) -> Dict[str, Tuple[float, float]]:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        
    def get_mean_std_vectors(self):
        if len(self.data_stats):
            mean_all = []
            std_all = []
            for key in MPC_LearnedContactPlanner.KEYS_ORDERED:
                if key == STATE_NAME:
                    mean, std = self.data_stats[key]
                    mean = mean[7:7+12]
                    std = std[7:7+12]
                else:
                    mean, std = self.data_stats[key]

                mean_all.extend(mean)
                std_all.extend(std)
                
            self.mean_input = np.array(mean_all)
            self.std_input = np.array(std_all) + 1e-12
            
    def normalize_inputs(self, input : np.array) -> np.array:
        if len(self.data_stats):
            input = (input - self.mean_input) / self.std_input
        
        return input
    
    def reset(self):
        """
        Reset controller.
        """
        self.i_cnt_replan = 0
        self.contact_plan_des = []
        self.full_length_contact_plan = []
        self.replanning = 0 # Replan contacts

        # MPC timings parameters
        self.sim_t = 0.0
        self.index = 0
        self.step = 0
        self.pln_ctr = 0
        self.horizon = int(self.replanning_time / self.sim_dt) # s

        self.diverged = False
        
        self.set_command()
        
    def set_goal(self, goal_id : np.ndarray) -> None:
        """ 
        Set the goal positions for each end effectors.
        goal_id contains the index of the goal stepping stones.
        """
        self.goal_locations = self.stones_env.positions[goal_id]
    
    def transform_points(self, b_T_W, points_w) -> np.ndarray:
        # Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_w.shape[0], 1))
        points_w_homogeneous = np.hstack((points_w, ones))
        # Apply the transformation matrix
        points_b_homogeneous = b_T_W @ points_w_homogeneous.T
        # Convert back to 3D coordinates
        points_b = points_b_homogeneous[:3, :].T
        return points_b
    
    def get_inputs(self, q, v, normalized : bool = True) -> np.ndarray:
        """
        Return model input data as np.array.
        """
        b_T_W = pin.XYZQUATToSE3(q[:7]).inverse()
        
        # Joint pos
        join_pos = q[7:7+12]

        # Goal location of feet (x, y, z) in base frame
        goal_location_b = self.transform_points(b_T_W, self.goal_locations).reshape(-1)

        # (vx, vy, vz, wx, wy, wz) velocities
        velocity = v[:6]

        # Current position of the legs (x, y, z) in base frame
        feet_pos_w = self.robot.get_foot_locations_world()
        feet_pos_b = self.transform_points(b_T_W, feet_pos_w).reshape(-1)

        # Get all box_location w.r.t to base
        stone_pose_b = self.transform_points(b_T_W, self.stones_env.positions).reshape(-1)
        stone_pose_b[self.stones_env.id_to_remove] = -1.

        input = np.hstack((
            goal_location_b,
            feet_pos_b,
            velocity,
            join_pos,
            stone_pose_b,
            ))
        
        input = self.normalize_inputs(input)
        
        return input
    
    def get_desired_contacts(self, q, v) -> np.ndarray:
        """
        Returns the desired contact positions for the <horizon>
        next timesteps of the MPC based on the desired contact plan.
        Should be called before the MPC is called.

        Returns:
            np.ndarray: Contact plan. Shape [H, 4, 3].
        """
        
        if self.replanning == 0:
            self.full_length_contact_plan = np.empty((0,4,3))

        mpc_contacts_w = []
        if self.goal_locations is not None:

            # Update the contact location only when robot is in contact
            if self.replan_contact():                
                input = self.get_inputs(q, v)
                input_array = torch.from_numpy(input).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # DDPM
                    if hasattr(self.model, "noise_scheduler"):
                        outputs_b = self.model.sample(
                            condition=input_array.reshape(1, -1, 3),
                            num_inference_steps=self.diffusion_steps
                            )
                        # CDCD
                        if hasattr(self.model.eps_model, "pointers"):
                            outputs_b = self.model.eps_model.selected
                            #max_probs = self.model.eps_model.max_probs.squeeze().detach().numpy()
                    else:
                        outputs_b = self.model(input_array)
                        
                outputs_b = outputs_b.cpu().numpy().reshape(8, 3)
                W_T_b = pin.XYZQUATToSE3(q[:7])
                outputs_W = self.transform_points(W_T_b, outputs_b)
                
                if self.project:
                    outputs_W = self.stones_env.get_closest_xy(outputs_W)[1]
                
                outputs_W = outputs_W.reshape(1, -1, 3)
                    
                jump1, jump2 = np.split(outputs_W, 2, axis=1)

                # Set full lenght contact plan
                if len(self.full_length_contact_plan) == 0:
                    # Set next 2 jumps
                    contact_plan = np.concatenate((jump1, jump2), axis=0)
                    repeat_contact_plan = np.repeat(contact_plan, self.gait_horizon, axis=0)
                    self.full_length_contact_plan = repeat_contact_plan
                    
                else:
                    # Add only jump2 to the contact plan
                    if self.replanning < 20:
                        repeat_contact_plan = np.repeat(jump1, self.gait_horizon, axis=0)
                    else:
                        repeat_contact_plan = np.repeat(jump2, self.gait_horizon, axis=0)
                        
                    self.full_length_contact_plan = np.concatenate(
                        (self.full_length_contact_plan, repeat_contact_plan),
                        axis=0)
                    
            mpc_contacts_w = self.full_length_contact_plan[self.replanning:self.replanning + 2*self.gait_horizon]

            # Update the desired velocity
            i = self.gait_horizon
            avg_position_next_cnt = np.mean(mpc_contacts_w[i], axis=0)
            self.v_des = np.round((avg_position_next_cnt - q[:3]) / self.gait_period, 2)
            self.v_des *= 1.4
            self.v_des[-1] = 0.
            
        return mpc_contacts_w