import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pinocchio as pin
import pickle

FILE_NAME = "data.npz"
MCTS_PERF_NAME = "mcts_perf.pkl"
STATE_NAME = "state"
GOAL_NAME = "state"
CONTACT_NAME = "contact_w"
TARGET_NAME = "target_w"
TARGET_ID_NAME = "target_id"
STONES_NAME = "stones_w"
    
class JumpDataset(Dataset):
    def __init__(self, experiment_dir, transform_to_base_frame=True):
        self.experiment_dir = experiment_dir
        self.transform_to_base_frame = transform_to_base_frame
        self.data = self.load_data()

    def load_data(self):
        states = []
        goals = []
        feet_contacts = []
        target_contacts = []
        target_contact_ids = []
        stones_pos_all = []
        N_samples = 0

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
                        
                    goal_id = self.load_goal(goal_dir)
                    if len(goal_id) == 0:
                        continue
                    
                    goal_locations_w = record_stones_pos_w[goal_id]

                    for i in range(len(record_state)):
                        states.append(record_state[i])
                        goals.append(goal_locations_w)
                        feet_contacts.append(record_feet_contact[i])
                        target_contacts.append(record_target_contact[i])
                        target_contact_ids.append(record_target_contact_id[i])
                        stones_pos_all.append(record_stones_pos_w)
                        N_samples += 1

        states = np.array(states)
        goals = np.array(goals)
        velocties = states[:, 7+12:7+12+6]
        feet_contacts = np.array(feet_contacts)
        target_contacts = np.array(target_contacts)
        target_contact_ids = np.array(target_contact_ids)
        stones_pos_all = np.array(stones_pos_all)

        if self.transform_to_base_frame:
            goals, feet_contacts, stones_pos_all, target_contacts = self.batch_transform_to_base_frame(
                states, goals, feet_contacts, stones_pos_all, target_contacts)

        data = {
            'velocities': torch.tensor(velocties, dtype=torch.float32).reshape(N_samples, -1),
            'goals': torch.tensor(goals, dtype=torch.float32).reshape(N_samples, -1),
            'feet_contacts': torch.tensor(feet_contacts, dtype=torch.float32).reshape(N_samples, -1),
            'stones_pos_all': torch.tensor(stones_pos_all, dtype=torch.float32).reshape(N_samples, -1),
            'target_contacts': torch.tensor(target_contacts, dtype=torch.float32).reshape(N_samples, -1),
            'target_contact_ids': torch.tensor(target_contact_ids, dtype=torch.int64).reshape(N_samples, -1)
        }

        return data
    
    def load_goal(self, goal_dir:str) -> np.ndarray:
        file_path = os.path.join(goal_dir, MCTS_PERF_NAME)
        goal_id = []
        if os.path.exists(file_path):
            # Saving a dictionary to a Pickle file
            with open(file_path, 'rb') as f:
                mcts_perf = pickle.load(f)
            goal_id = mcts_perf["goal"]
        
        return goal_id

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
        return len(self.data['velocities'])

    def __getitem__(self, idx):
        goal_contact_base = self.data['goals'][idx]
        feet_contact_base = self.data['feet_contacts'][idx]
        velocities = self.data['velocities'][idx]
        stones_pos_base = self.data['stones_pos_all'][idx]
        target_locations = self.data['target_contacts'][idx].reshape(-1, 3) if self.transform_to_base_frame else self.data['target_contact_ids'][idx]

        state_goal_conditionning = torch.cat((goal_contact_base, feet_contact_base, velocities, stones_pos_base)).reshape(-1, 3)

        batch = {
            "data": target_locations,
            "condition": state_goal_conditionning,
        }
        
        return batch

def shuffle_collate(batch):
    condition = torch.stack([d["condition"] for d in batch], dim=0)
    data = torch.stack([d["data"] for d in batch], dim=0)
    B, _, C = data.shape

    condition = condition.reshape(B, -1, C)
    n_state = 10
    n_boxes = condition.shape[1] - n_state
    shuffle_indices = torch.hstack((torch.arange(n_state), torch.randperm(n_boxes) + n_state)).unsqueeze(-1).unsqueeze(0) # Shuffle boxes

    # Shuffle inputs along dimension 1
    shuffled_condition = torch.take_along_dim(condition, shuffle_indices, dim=1).reshape(B, -1)
    
    shuffled_batch = {
        "data" : data,
        "condition" : shuffled_condition,
    }
    
    return shuffled_batch

def get_dataloaders(data_dir, dataset, batch_size, return_index:bool=False, augmentation:bool=False, shuffle:bool=False, noise_level:float=1.e-3, train_only=False):
    train_data_path = os.path.join(data_dir, dataset, "train")
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(data_dir, dataset)
    train_dataset = JumpDataset(train_data_path)


    test_data_path = os.path.join(data_dir, dataset, "test")
    if os.path.exists(test_data_path) and not(train_only):
        test_dataset = JumpDataset(test_data_path)
    else:
        test_dataset = None

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=shuffle_collate if shuffle else None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset != None else None
    
    batch = next(iter(train_dataloader))

    print("Train batch shape:")
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_dataloaders("/home/atari_ws/data/", "", 32)
    print(len(train_dataloader.dataset))