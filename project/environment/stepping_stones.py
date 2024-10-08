import numpy as np
import os
from typing import Tuple

class SteppingStonesEnv:
    DEFAULT_FILE_NAME = "env_desc.npz"
    DEFAULT_STONE_SHAPE = "box"               # box or cylinder
    DEFAULT_STONE_HEIGHT = 0.1                # m
    
    def __init__(self,
                 grid_size: Tuple[int, int] = (9, 9),
                 spacing: Tuple[float, float] = (0.19, 0.147),
                 size_ratio: Tuple[float, float] = (0.75, 0.75),
                 randomize_pos_ratio: float = 0.,
                 randomize_height_ratio: float = 0.,
                 N_to_remove: int = 0,
                 **kwargs) -> None:
        """
        Define stepping stones locations on a grid. 

        Args:
            - grid_size (Tuple[int, int], optional): Number of stepping stones node (x, y).
            - spacing (Tuple[float, float], optional): Spacing of the center of the stones (x, y).
            - size_ratio (Tuple[float, float], optional): Size ratio of the stepping 
            stone and the spacing.
            size_ratio[0] * spacing and size_ratio[1] * spacing. Defaults to False.
            - randomize_pos (float, optional): Randomize stepping stone location within it's area 
            without collision. Ratio to the max displacement. Defaults to 0, no displacement.
            - randomize_height_ratio (float, optional): Randomize height between [(1-ratio)*h, (1+ratio)*h].
            - N_to_remove (int, optional): Number of stones to remove.
        """
        self.grid_size = grid_size
        self.randomize_pos_ratio = randomize_pos_ratio
        self.spacing = list(spacing)
        self.size_ratio = list(size_ratio)
        self.randomize_height_ratio = randomize_height_ratio
        self.N_to_remove = N_to_remove
        
        # Optional args
        self.shape = None
        self.height = None
        optional_args = {
            "shape" : SteppingStonesEnv.DEFAULT_STONE_SHAPE,
            "height" : SteppingStonesEnv.DEFAULT_STONE_HEIGHT,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        self.I = self.grid_size[0]
        self.J = self.grid_size[1]
        self.N = self.I * self.J
        self.id_to_remove = np.array([], dtype=np.int32)
        self.init_stones()
        
    def init_stones(self) -> None:
        """ 
        Init stones positions.
        """
        self._init_center_location()  
        self._init_size()
        self._randomize_height()
        self._randomize_center_location()
        
    def _init_center_location(self) -> None:
        """
        Initialize the center locations of the stepping stones.
        """        
        ix = np.arange(self.I) - self.I // 2
        iy = np.arange(self.J) - self.J // 2
        z = np.full(((self.N, 1)), self.height)

        nodes_xy = np.dstack(np.meshgrid(ix, iy)).reshape(-1, 2)
        stepping_stones_xy = nodes_xy * np.array([self.spacing])
        self.positions = np.hstack((stepping_stones_xy, z))

    def _randomize_height(self) -> None:
        """
        Randomize the height of the stones.
        """
        self.positions[:, -1] = self.height + (np.random.rand(self.N) - 0.5) * 2 * self.randomize_height_ratio * self.height
        
    def _init_size(self) -> None:
        """
        Init the size of the stepping stones.
        """
        size_ratio = np.random.uniform(
            low=self.size_ratio[0],
            high=self.size_ratio[1],
            size=self.N
            )
        self.size = size_ratio * min(self.spacing)
        
    def _randomize_center_location(self) -> None:
        """
        Randomize the center of the stepping stones locations.
        """
        max_displacement_x = (self.spacing[0] - self.size) / 2.
        max_displacement_y = (self.spacing[1] - self.size) / 2.
        
        dx = np.random.uniform(-1., 1., self.N) * max_displacement_x * self.randomize_pos_ratio
        dy = np.random.uniform(-1., 1., self.N) * max_displacement_y * self.randomize_pos_ratio

        self.positions[:, 0] += dx
        self.positions[:, 1] += dy
        
    def remove_random(self, N_to_remove: int = -1, keep: list[int] = []) -> None:
        """
        Randomly remove stepping stones.
        
        Args:
            N_to_remove (int): Number of box to remove.
            keep (list[int]): id of the stepping stones to keep
        """
        # 0 probability for id in keep
        probs = np.ones((self.N,))
        probs[keep] = 0.
        probs /= np.sum(probs)
        
        if N_to_remove == -1:
            N_to_remove = self.N_to_remove
        
        self.id_to_remove = np.random.choice(self.N, N_to_remove, replace=False, p=probs)
        self.positions[self.id_to_remove, -1] = -10.
            
    def get_closest(self, positions_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the indices and positions of the stepping stones closest to 
        each position in <position_xyz>. 

        Args:
            positions_xyz (np.ndarray): array of N 3D positions [N, 3].
        """
        
        # Squared distance
        diffs = self.positions[:, np.newaxis, :] - positions_xyz[np.newaxis, :, :]
        d_squared = np.sum(diffs**2, axis=-1)

        # Find the indices of the closest points
        closest_indices = np.argmin(d_squared, axis=0)
        
        # Extract the closest points from stepping stones
        closest_points = self.positions[closest_indices]

        return closest_indices, closest_points
    
            
    def get_closest_xy(self, positions_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the indices and positions of the stepping stones closest to 
        each position in <position_xyz>. 

        Args:
            positions_xy (np.ndarray): array of N 3D positions [N, 3].
        """
        
        # Squared distance
        diffs = self.positions[:, np.newaxis, :2] - positions_xy[np.newaxis, :, :2]
        d_squared = np.sum(diffs**2, axis=-1)

        # Find the indices of the closest points
        closest_indices = np.argmin(d_squared, axis=0)
        
        # Extract the closest points from stepping stones
        closest_points = self.positions[closest_indices]

        return closest_indices, closest_points
    
    def set_start_position(self, start_pos: np.array) -> np.ndarray:
        """
        Set closest x, y of stepping stones of the start positions
        to x, y of start positions.

        Args:
            start_pos (np.array): Start positions. Shape [N, 3].
        Returns:
            np.ndarray: stepping stones closest to start positions.
        """
        id_closest_to_start, _ = self.get_closest(start_pos)
        self.positions[id_closest_to_start, :2] = start_pos[:, :2] 
        self.positions[id_closest_to_start, 2] = self.height
        self.size[id_closest_to_start] = (self.size_ratio[0] + self.size_ratio[1]) / 2. * min(self.spacing)

        return self.positions[id_closest_to_start]
    
    def pick_random(self, positions_xyz: np.ndarray, d_max : float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pick random stepping stones around given positions at a maximum distance of d_max.

        Args:
            positions_xyz (np.ndarray): array of N 3D positions [N, 3].
            d_max (float): maximum distance to consider for picking stones.

        Returns:
            Tuple[np.ndarray, np.ndarray]: id [N], positions [N, 3]
        """
        # Squared distance
        diffs = self.positions[:, np.newaxis, :] - positions_xyz[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))

        # Init
        N = len(positions_xyz)
        chosen_indices = np.zeros(N, dtype=np.int32)
        chosen_positions = np.zeros_like(positions_xyz, dtype=np.float32)
        
        for i in range(N):
            # Filter based on d_max
            within_d_max = dist[:, i] <= d_max
            # Get valid indices
            valid_indices = np.where(within_d_max)
            
            if len(valid_indices[0]) == 0:
                raise ValueError("No positions found within the specified distance.")

            id = np.random.choice(valid_indices[0], replace=False)
            pos = self.positions[id]
            
            chosen_indices[i] = id
            chosen_positions[i] = pos

        return chosen_indices, chosen_positions
    
    def save(self, save_dir: str) -> None:
        """
        Save the environment's state as a npz file.
        """
        save_path = os.path.join(save_dir, SteppingStonesEnv.DEFAULT_FILE_NAME)
        np.savez(
            save_path,
            grid_size=self.grid_size,
            randomize_pos_ratio=self.randomize_pos_ratio,
            spacing=self.spacing,
            size_ratio=self.size_ratio,
            randomize_height_ratio=self.randomize_height_ratio, 
            shape=self.shape,
            height=self.height,
            positions=self.positions,
            size=self.size,
            id_to_remove=self.id_to_remove
            )
        
    @staticmethod
    def load(env_dir: str) -> 'SteppingStonesEnv':
        """
        Load the environment's state from a npz file.
        """
        path = os.path.join(env_dir, SteppingStonesEnv.DEFAULT_FILE_NAME)
        data = np.load(path)
        env = SteppingStonesEnv(grid_size=tuple(data['grid_size']),
                                spacing=tuple(data['spacing']),
                                size_ratio=tuple(data['size_ratio']),
                                randomize_pos_ratio=data['randomize_pos_ratio'],
                                randomize_height_ratio=data['randomize_height_ratio'],
                                shape=data['shape'],
                                height=data['height'])
        
        env.positions = data['positions']
        env.size = data['size']
        env.id_to_remove = data['id_to_remove']
        
        return env