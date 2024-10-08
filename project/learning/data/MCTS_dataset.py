import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pinocchio as pin
import pickle
from typing import List, Tuple, Dict

FILE_NAME = "data.npz"
STATE_NAME = "state"
GOAL_NAME = "goal_w"
FEET_POS_NAME = "contact_w"
TARGET_NAME = "target_w"
TARGET_ID_NAME = "target_id"
CONTACT_NAME = "stones_w"
STONES_SHUFFLE = "shuffle_stones_w"
TARGET_ID_SHUFFLE = "target_id_shuffle"
NORMALIZATION_STATS = "normalization_stats.pkl"

class MCTSDataset(Dataset):
    
    VARIABLE_TO_NORMALIZE = [
        "goals_b",
        "feet_pos_b",
        "velocities",
        "qj",
        "cnt_pos_b",
    ]
    
    def __init__(self,
                 exp_dir: str,
                 normalize: bool = False,
                 stats_file: str = None,
                 shuffle_period: int = 0,
                 transform_to_base_frame: bool = True,
                 return_index : bool = False,
                 ddpm : bool = False,
                 **kwargs):
        """
        A dataset class to load and preprocess MCTS data for training.
        
        Args:
            experiment_dir (str): Directory with experiment data.
            normalize (bool): Apply normalization to the data.
            stats_file (str): Path to a file to save or load normalization stats.
            shuffle (bool): Shuffle all contact locations.
            shuffle_period (int): Shuffle every <shuffle_period>. 0 if no shuffle.
            transform_to_base_frame (bool): Transform the data to the robot's base frame.
            return_index (bool): Return index of the target in the contact sequence instead of location.
            ddpm (bool): data for diffusion model.
        """
        self.experiment_dir = exp_dir
        self.normalize = normalize
        self.stats_file = stats_file
        self.transform_to_base_frame = transform_to_base_frame
        self.current_step = 0
        self.return_index = return_index
        self.ddpm = ddpm
        
        # Load the dataset
        self.data = self.load_data()
        self.shuffle_period = shuffle_period * self.N_samples

        # Apply normalization if requested
        if self.normalize:
            self.stats = self.load_or_compute_stats()
            self.apply_normalization()

    def load_data(self) -> Dict[str, torch.Tensor]:
        """Load data from multiple experiments and environments."""
        states, goals_w, feet_pos_w, target_contacts_w = [], [], [], []
        target_contact_ids, cnt_pos_all_w = [], []
        self.N_samples = 0

        for env_name in os.listdir(self.experiment_dir):
            env_dir = os.path.join(self.experiment_dir, env_name)
            if not os.path.isdir(env_dir):
                continue

            for goal_name in os.listdir(env_dir):
                goal_dir = os.path.join(env_dir, goal_name)
                if not os.path.isdir(goal_dir):
                    continue

                file_path = os.path.join(goal_dir, FILE_NAME)
                if os.path.isfile(file_path):
                    with np.load(file_path) as f:
                        record_state = f[STATE_NAME]
                        record_feet_contact = f[FEET_POS_NAME]
                        record_target_contact = f[TARGET_NAME]
                        record_target_contact_id = f[TARGET_ID_NAME]
                        record_cnt_pos_w = f[CONTACT_NAME]
                        goal_pos_w = f[GOAL_NAME]
                    
                    if len(goal_pos_w) == 0:
                        continue
                    
                    N = len(record_state)
                    for i in range(N):
                        goals_w.append(goal_pos_w)
                        cnt_pos_all_w.append(record_cnt_pos_w)
                        states.append(record_state[i])
                        feet_pos_w.append(record_feet_contact[i])
                        target_contacts_w.append(record_target_contact[i])
                        target_contact_ids.append(record_target_contact_id[i])
                        self.N_samples += 1

        self.N_cnt = len(record_cnt_pos_w)
        # Convert lists to numpy arrays
        self.states = torch.tensor(np.array(states)).float()
        self.velocities = self.states[:, 7+12:7+12+6]  # Extract velocity data
        self.qj = self.states[:, 7:7+12]  # Extract velocity data
        self.cnt_pos_all_w = np.array(cnt_pos_all_w)
        self.goals_w = np.array(goals_w)
        self.feet_pos_w = np.array(feet_pos_w)
        self.target_contacts_w = np.array(target_contacts_w)
        self.target_contact_ids = torch.tensor(np.array(target_contact_ids)).long()
        self._batch_world_to_base_frame()
        
        # Setup copy for shuffled tensor
        self.target_contact_ids_ = torch.clone(self.target_contact_ids)
        self.cnt_pos_b_ = torch.clone(self.cnt_pos_b).reshape(self.N_samples, -1, 3)
        
        # Normalize the data if requested
        if self.normalize:
            data_dir, _ = os.path.split(data_dir)
            self.normalization_file_path = os.path.join(data_dir, MCTSDataset.NORMALIZATION_FILE)
            # Compute normalization parameters and save them
            self.mean_std = self.compute_normalization()
            
            if self.normalization_file_path:
                self.save_normalization(self.normalization_file_path, self.mean_std)
    
            # Apply normalization to the input data
            self.apply_normalization()
        
        del self.feet_pos_w, self.target_contacts_w, self.goals_w
        
    def compute_normalization(self):
        """
        Compute the mean and standard deviation of the data for normalization.

        Returns:
            dict: Dictionary containing the mean and standard deviation for each feature.
        """
        mean_stats = {}
        std_stats = {}
        for var_name in MCTSDataset.VARIABLE_TO_NORMALIZE:
            if var_name in self.__dict__:
                # Dynamically access the variable by its name
                exec(f"mean_stats['{var_name}'] = self.{var_name}.reshape(self.N, -1).mean(dim=0)")
                exec(f"std_stats['{var_name}'] = self.{var_name}.reshape(self.N, -1).std(dim=0)")
            else:
                raise ValueError(f"Unknown variable: {var_name}")
            
        normalization_stats = {"mean" : mean_stats, "std" : std_stats}
        return normalization_stats

    def apply_normalization(self):
        """
        Normalize the input data using the precomputed mean and std.
        """
        mean_dict = self.mean_std["mean"]
        std_dict = self.mean_std["std"]
        
        for var_name in MCTSDataset.VARIABLE_TO_NORMALIZE:
            if var_name in self.__dict__:
                # Dynamically access the variable by its name
                mean, std = mean_dict[var_name], std_dict[var_name]
                exec(f"self.{var_name} = self.{var_name}.reshape(self.N, -1)")
                exec(f"self.{var_name} = (self.{var_name} - mean) / (std + 1.0e-8)")
            else:
                raise ValueError(f"Unknown variable: {var_name}")

    def save_normalization(self, path: str, mean_std: dict):
        """
        Save the normalization parameters (mean and std) to a file.

        Args:
            path (str): Path to save the normalization parameters.
            mean_std (dict): Dictionary containing the mean and std tensors.
        """
        with open(path, 'wb') as f:
            pickle.dump(mean_std, f)

    def load_normalization(self, path: str) -> dict:
        """
        Load normalization parameters (mean and std) from a file.

        Args:
            path (str): Path to load the normalization parameters from.

        Returns:
            dict: Dictionary containing the mean and std tensors.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def batch_transform_to_base_frame(q_batch, points_w_batch):
        """
        Batch transform points from world frame to the robot's base frame for multiple states.

        Args:
            q_batch (np.ndarray): Batch of robot states, shape [N, 7] where each row is [x, y, z, qx, qy, qz, qw].
            points_w_batch (np.ndarray): Batch of points in world frame, shape [N, M, 3] where M is the number of points.

        Returns:
            np.ndarray: Batch of points transformed to base frame, shape [N, M, 3].
        """
        # Compute the inverse transformation matrices from world to base frame for each q in the batch
        B_T_W_batch = np.array([pin.XYZQUATToSE3(q[:7]).inverse().homogeneous for q in q_batch])
        
        # Homogeneous transformation: Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_w_batch.shape[0], points_w_batch.shape[1], 1))
        points_w_homogeneous = np.concatenate((points_w_batch, ones), axis=-1)  # Shape [N, M, 4]

        # Apply the batch transformation: B_T_W_batch @ points_w_homogeneous
        points_b_homogeneous = np.einsum('nij,nmj->nmi', B_T_W_batch, points_w_homogeneous)
        
        # Convert back to 3D coordinates by dropping the homogeneous coordinate
        points_b = points_b_homogeneous[:, :, :3]
        
        return points_b
    
    def _batch_world_to_base_frame(self):
        """
        Process all data from world frame to base frame using batch matrix operations.
        """
        # Concatenate all points in world frame
        points_w_batch = np.concatenate((
            self.goals_w.reshape(-1, 4, 3),                  # Goal locations
            self.target_contacts_w.reshape(-1, 8, 3),        # Target contacts
            self.feet_pos_w.reshape(-1, 4, 3),               # Feet positions
            self.cnt_pos_all_w.reshape(-1, self.N_cnt, 3),        # Contact positions
        ), axis=1)

        # Apply batch transformation
        points_b_batch = self.batch_transform_to_base_frame(self.states[:, :7].numpy(), points_w_batch)

        # Split the transformed points into their respective tensors
        self.goals_b, self.target_contact_b, self.feet_pos_b, self.cnt_pos_b = torch.split(
            torch.from_numpy(points_b_batch).float(),
            [4, 8, 4, self.N_cnt],
            dim=1,
        )
        self.goals_b = self.goals_b.reshape(self.N_samples, 4*3)
        if self.ddpm:
            self.target_contact_b = self.target_contact_b.reshape(self.N_samples, -1,3)
        else:
            self.target_contact_b = self.target_contact_b.reshape(self.N_samples, -1)
        self.feet_pos_b = self.feet_pos_b.reshape(self.N_samples, -1)
        self.cnt_pos_b = self.cnt_pos_b.reshape(self.N_samples, -1)

    def shuffle_all_contacts(self):
        """Shuffle the contact locations for the entire dataset."""
        for i, (cnt_pos_b, target_index) in enumerate(zip(self.cnt_pos_b, self.target_contact_ids)):
            cnt_pos_b = cnt_pos_b.reshape(-1, 3)

            # Get unique indices
            unique_target_index, inverse_index = torch.unique(target_index, sorted=False, return_inverse=True)
            T = len(unique_target_index)
            N_cnt = len(cnt_pos_b)
            
            # Shuffle target indices based on permutation
            permutation = torch.rand(N_cnt).argsort(dim=0)
            shuffled_target_id, shuffled_remaining_id = torch.split(permutation, [T, N_cnt - T])
            shuffled_target_id = shuffled_target_id[inverse_index]
        
            # Shuffle targets and remaining indices
            # Precompute remaining indices for efficient access
            all_indices = torch.arange(N_cnt).long()
            mask = torch.ones_like(all_indices, dtype=torch.bool)
            mask.scatter_(0, target_index.clone().long(), False)
            remaining_indices = all_indices[mask]

            # Shuffle contact locations within the dataset itself
            self.cnt_pos_b_[i, shuffled_target_id, :] = cnt_pos_b[target_index, :]
            self.cnt_pos_b_[i, shuffled_remaining_id, :] = cnt_pos_b[remaining_indices, :]
            self.target_contact_ids_[i] = shuffled_target_id

    def maybe_shuffle_all_contacts(self):
        """Shuffle contacts periodically, based on `shuffle_period`."""
        self.current_step += 1
        if self.shuffle_period > 0 and self.current_step % self.shuffle_period == 0:
            self.shuffle_all_contacts()
            self.current_step = 0

    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        """Get a single sample by index."""
        self.maybe_shuffle_all_contacts()
        
        state_goal_conditioning = torch.cat((
            self.goals_b[idx],
            self.feet_pos_b[idx],
            self.velocities[idx],
            self.qj[idx],
            self.cnt_pos_b_[idx].reshape(-1)
        ))

        # Diffusion model
        if self.ddpm:
            return {
                "data": self.target_contact_b[idx],
                "indices": self.target_contact_ids_,
                "condition": state_goal_conditioning,
            }
        # Supervised learning
        else:
            return {
                "input": state_goal_conditioning,
                "target": self.target_contact_b[idx],
            }