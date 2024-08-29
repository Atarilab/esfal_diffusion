import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pinocchio as pin
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

FILE_NAME = "data.npz"
MCTS_PERF_NAME = "mcts_perf.pkl"
STATE_NAME = "state"
VELOCITY_NAME = "velocity"
GOAL_NAME = "goal_w"
CONTACT_NAME = "contact_w"
TARGET_NAME = "target_w"
TARGET_ID_NAME = "target_id"
STONES_NAME = "stones_w"
NORMALIZATION_STATS = "normalization_stats.pkl"

import numpy as np
import copy
import pinocchio as pin

def apply_symmetry_3d_points(a : np.array, x : int, y : int):
    '''
    apply symmetry to an array of 3D points.
    a [N, 3]
    '''
    a_copy = np.copy(a)
    if len(a.shape) == 2:
        if x < 0 and y < 0:
            a_copy[:, :2] = -a[:, :2]
            
        elif x < 0:
            a_copy[:, 0] = -a[:, 0]

        elif y < 0:
            a_copy[:, 1] = -a[:, 1]
            
    elif len(a.shape) == 1:
        if x < 0 and y < 0:
            a_copy[:2] = -a[:2]
            
        elif x < 0:
            a_copy[0] = -a[0]

        elif y < 0:
            a_copy[1] = -a[1]  
                  
    return a_copy

# FR  -0.0008484878775420449, 0.6875545803138652, -1.331231761101011,
# FL  -0.0044557643132792915, 0.6926241261462901, -1.3313565227202124,
# RR  0.005418005992576224, -0.6904568346955275, 1.330043456714709,
# RL  0.0019115460586124494, -0.6892011722464748, 1.3372704771426844,

ID_SWITCH_X = [3,4,5,0,1,2,9,10,11,6,7,8]
ID_SWITCH_Y = [6,7,8,9,10,11,0,1,2,3,4,5]
ID_SWITCH_XY = [9,10,11,6,7,8,3,4,5,0,1,2]

def apply_symmetry_joints(q : np.array, x : int = 0, y : int = 0):
    '''
    apply symmetry to a joint configuration.
    q : [12]
    [FR, FL, RR, RL]
    '''
    # Switch left and right legs and back front
    q_copy = np.copy(q)
    
    if x < 0 and y < 0:
        q_copy = -q[ID_SWITCH_XY]
        
    # Switch left and right legs
    elif x < 0:
        q_copy = q[ID_SWITCH_X]

    # Switch front and back legs
    elif y < 0:
        q_copy = -q[ID_SWITCH_Y]
        
    return q_copy

def get_symmetric_state(q : np.array, x : int = 0, y : int = 0):
    '''
    apply symmetry to full state configuration.
    q : [3 + 7 + 12 + 6 + 12]
    '''
    q_copy = np.copy(q)

    if x > 0 and y > 0:
        return q_copy
    
    # position is the same
    # symmetry orientation as rpy
    R = pin.Quaternion(np.array(q_copy[3:7])).toRotationMatrix()
    rpy_vector = pin.rpy.matrixToRpy(R)
    rpy_vector = apply_symmetry_3d_points(rpy_vector, x, y)
    R = pin.rpy.rpyToMatrix(rpy_vector)
    q = pin.Quaternion(R)
    q_copy[3:7] = np.array([q.x, q.y, q.z, q.w])
    # joints position
    q_copy[7:7+12] = apply_symmetry_joints(q_copy[7:7+12], x, y)
    # velocities in world frame
    q_copy[19:22] = apply_symmetry_3d_points(q_copy[19:22], x, y)
    q_copy[22:25] = apply_symmetry_3d_points(q_copy[22:25], x, y)
    # joint velocities
    q_copy[-12:] = apply_symmetry_joints(q_copy[-12:], x, y)

    return q_copy
    
class JumpDataset(Dataset):
    def __init__(self,
                 experiment_dir : str,
                 normalize : bool = False,
                 stats_file : bool = None,
                 augmentation : bool = False,
                 return_index : bool = False):
        self.experiment_dir = experiment_dir
        self.normalize = normalize
        self.stats_file = stats_file
        self.augmentation = augmentation
        self.return_index = return_index
        self.sym = [1]
        if self.augmentation:
            self.sym = [-1, 1]
        
        self.data = self.load_data()
        if self.normalize:
            if self.stats_file and os.path.exists(self.stats_file):
                self.stats = self.load_stats(self.stats_file)
            else:
                self.stats = self.compute_stats()
                if self.stats_file:
                    self.save_stats(self.stats_file)
            self.apply_normalization()

    def load_data(self):
        states = []
        goals = []
        feet_contacts = []
        target_contacts = []
        target_contact_ids = []
        stones_pos_all = []
        N_samples = 0

        s = 0
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
                        record_feet_contact = f[CONTACT_NAME]
                        record_target_contact = f[TARGET_NAME]
                        record_target_contact_id = f[TARGET_ID_NAME]
                        record_stones_pos_w = f[STONES_NAME]
                        goal_pos_w = f[GOAL_NAME]
                    
                    if len(goal_pos_w) == 0:
                        continue
                    
                    start_pos = record_target_contact_id[0][:4]
                    
                    for i in range(len(record_state)):
                        s += 1
                        if (record_target_contact_id[i][4:] == start_pos).all():
                            continue

                        states.append(record_state[i])
                        goals.append(goal_pos_w)
                        feet_contacts.append(record_feet_contact[i])
                        target_contacts.append(record_target_contact[i])
                        stones_pos_all.append(record_stones_pos_w)
                        target_contact_ids.append(record_target_contact_id[i])
                        N_samples += 1

        states = np.array(states)
        goals = np.array(goals)
        velocities = states[:, 7+12:7+12+6]
        feet_contacts = np.array(feet_contacts)
        target_contacts = np.array(target_contacts)
        target_contact_ids = np.array(target_contact_ids)
        stones_pos_all = np.array(stones_pos_all)

        goals, feet_contacts, stones_pos_all, target_contacts = self.batch_transform_to_base_frame(
            states, goals, feet_contacts, stones_pos_all, target_contacts
            )

        data = {
            VELOCITY_NAME: torch.tensor(velocities, dtype=torch.float32).reshape(N_samples, -1),
            STATE_NAME: torch.tensor(states, dtype=torch.float32).reshape(N_samples, -1),
            GOAL_NAME: torch.tensor(goals, dtype=torch.float32).reshape(N_samples, -1),
            CONTACT_NAME: torch.tensor(feet_contacts, dtype=torch.float32).reshape(N_samples, -1),
            STONES_NAME: torch.tensor(stones_pos_all, dtype=torch.float32).reshape(N_samples, -1),
            TARGET_NAME: torch.tensor(target_contacts, dtype=torch.float32).reshape(N_samples, -1),
            TARGET_ID_NAME: torch.tensor(target_contact_ids, dtype=torch.int64).reshape(N_samples, -1)
        }

        return data

    def compute_stats(self) -> Dict[str, Tuple[float, float]]:
        stats = {}
        for key in self.data:
            if "target" not in key:
                data = self.data[key].numpy()
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                stats[key] = (mean, std)
        return stats

    def save_stats(self, stats_file: str):
        with open(stats_file, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, stats_file: str) -> Dict[str, Tuple[float, float]]:
        with open(stats_file, 'rb') as f:
            return pickle.load(f)

    def apply_normalization(self):
        for key in self.data:
            if "target" not in key:
                mean, std = self.stats[key]
                self.data[key] = (self.data[key] - mean) / std

    def batch_transform_to_base_frame(self, states, goals, feet_contacts, stones_pos_all, target_contacts):
        transformed_goals = []
        transformed_feet_contacts = []
        transformed_stones_pos_all = []
        transformed_target_contacts = []

        for i in range(states.shape[0]):
            b_T_W = pin.XYZQUATToSE3(states[i, :7]).inverse()
            transformed_goals.append(self.transform_points(b_T_W, goals[i]))
            transformed_feet_contacts.append(self.transform_points(b_T_W, feet_contacts[i]))
            transformed_stones_pos_all.append(self.transform_points(b_T_W, stones_pos_all[i]))
            transformed_target_contacts.append(self.transform_points(b_T_W, target_contacts[i]))

        return np.array(transformed_goals), np.array(transformed_feet_contacts), np.array(transformed_stones_pos_all), np.array(transformed_target_contacts)

    def transform_points(self, b_T_W, points_w):
        # Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_w.shape[0], 1))
        points_w_homogeneous = np.hstack((points_w, ones))
        # Apply the transformation matrix
        points_b_homogeneous = b_T_W @ points_w_homogeneous.T
        # Convert back to 3D coordinates
        points_b = points_b_homogeneous[:3, :].T
        return points_b

    def __len__(self):
        return len(self.data[VELOCITY_NAME])

    def __getitem__(self, idx):
        join_pos = self.data[STATE_NAME][idx, 7:7+12]
        goal_contact_base = self.data[GOAL_NAME][idx]
        feet_contact_base = self.data[CONTACT_NAME][idx]
        velocities = self.data[VELOCITY_NAME][idx]
        stones_pos_base = self.data[STONES_NAME][idx]
        target_locations = self.data[TARGET_NAME][idx].reshape(-1, 3)
        target_index = self.data[TARGET_ID_NAME][idx]

        state_goal_conditioning = torch.cat((goal_contact_base, feet_contact_base, velocities, join_pos, stones_pos_base)).reshape(-1, 3)

        batch = {
            "data": target_locations,
            "condition": state_goal_conditioning,
            "index" : target_index,
        }
        return batch
    

def create_batched_index_tensor(N, exclude_batch):
    B, T = exclude_batch.shape  # Get the batch size (B) and number of exclusions per batch (T)

    # Create a tensor of all indices from 0 to N-1, and expand it to match the batch size
    all_indices = torch.arange(N).unsqueeze(0).expand(B, N)  # Shape: [B, N]
    
    # Create a mask tensor initialized to True
    mask = torch.ones((B, N), dtype=torch.bool)
    
    # Mark the excluded indices as False
    mask.scatter_(1, exclude_batch, False)
    
    # Apply the mask to get the desired indices
    valid_indices = torch.masked_select(all_indices, mask).view(B, -1)
    
    return valid_indices

def shuffle_collate(batch):
    condition = torch.stack([d["condition"] for d in batch], dim=0)
    data = torch.stack([d["data"] for d in batch], dim=0)
    target_indices = torch.stack([d["index"] for d in batch], dim=0)
    B = len(data)
    condition = condition.reshape(B, -1, 3)
    N_state = 14
    N = condition.shape[1] - N_state
    T = target_indices.shape[-1]
    state, cnt_locations = torch.split(condition, [N_state, N], dim=1)
    
    # Sample new target indices
    permutation = torch.rand(B, N).argsort (dim = 1) 

    # Split the permutation into targets and remaining indices
    shuffled_target_id, shuffled_remaining_id = torch.split(permutation, [T, N - T], dim=1)

    # Compute remaining indices (that are not targets)
    all_remaining_id = create_batched_index_tensor(N, exclude_batch=shuffled_target_id)

    # Initialize the shuffled contact locations tensor
    shuffled_cnt_locations = torch.empty_like(cnt_locations)

    # Assign the target contact locations to their shuffled positions
    shuffled_cnt_locations.scatter_(1, shuffled_target_id.unsqueeze(-1).expand(-1, -1, 3), cnt_locations.gather(1, target_indices.unsqueeze(-1).expand(-1, -1, 3)))

    # Assign the remaining contact locations to the remaining shuffled positions
    shuffled_cnt_locations.scatter_(1, shuffled_remaining_id.unsqueeze(-1).expand(-1, -1, 3), cnt_locations.gather(1, all_remaining_id.unsqueeze(-1).expand(-1, -1, 3)))

    # Add the state data
    shuffled_condition = torch.cat(
        (state, shuffled_cnt_locations), dim=1
    )
    
    shuffled_batch = {
        "data": data,
        "condition": shuffled_condition,
        "index" : shuffled_target_id,
    }

    return shuffled_batch

def get_dataloaders(data_dir, 
                    dataset, batch_size, 
                    return_index:bool=False, 
                    augmentation:bool=False, 
                    shuffle:bool=True, 
                    train_only=False, 
                    normalize=False):
    
    train_data_path = os.path.join(data_dir, dataset, "train")
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(data_dir, dataset)
        
    train_dataset = JumpDataset(
        train_data_path,
        normalize=normalize,
        augmentation=augmentation,
        return_index=return_index
        )

    test_data_path = os.path.join(data_dir, dataset, "test")
    stats_file = os.path.join(data_dir, dataset, NORMALIZATION_STATS)
    if os.path.exists(test_data_path) and not(train_only):
        test_dataset = JumpDataset(test_data_path, normalize=normalize, stats_file=stats_file)
    else:
        test_dataset = None

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=shuffle_collate if shuffle else None, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 8) if test_dataset != None else None
    
    batch = next(iter(train_dataloader))

    print("Number of samples:")
    print("Train:", len(train_dataset))
    if test_dataset != None: print("Test:", len(test_dataset))
    print("Train batch shape:")
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_dataloader, test_dataloader


def plot_histograms(data, feature_name):
    """
    Plot histograms for the given feature.
    Args:
        data (torch.Tensor): The data to plot histograms for.
        feature_name (str): The name of the feature.
    """
    data_np = data.numpy()
    plt.figure(figsize=(12, 6))
    for i in range(data_np.shape[1]):
        plt.subplot(data_np.shape[1] // 3, 3, i + 1)
        plt.hist(data_np[:, i], bins=30, alpha=0.75, edgecolor='black')
        plt.title(f'{feature_name} {i + 1}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the dataset
    experiment_dir = "/home/atari_ws/data/record_/train"
    dataset = JumpDataset(experiment_dir)

    # Plot histograms for each feature
    plot_histograms(dataset.data['target_contacts'][:, :12], 'Target Contacts')
    plot_histograms(dataset.data['target_contacts'][:, 12:], 'Target Contacts')
    plot_histograms(dataset.data['velocities'], 'Velocities')
    plot_histograms(dataset.data['goals'], 'Goals')
    plot_histograms(dataset.data['feet_contacts'], 'Feet Contacts')
    
    experiment_dir = "/home/atari_ws/data/randomized_positions"
    train_dataloader, test_dataloader = get_dataloaders(experiment_dir, "", 32)
    print(len(train_dataloader.dataset))
    batch = next(iter(train_dataloader))
    print(batch["condition"][10])
    print(batch["data"][10])