
import copy
from typing import Any, Callable, Tuple
import numpy as np
import pybullet
from environment.stepping_stones import SteppingStonesEnv
from py_pin_wrapper.simulator import Simulator
from mpc_controller.bicon_mpc import BiConMPC
from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from py_pin_wrapper.abstract.data_recorder import DataRecorderAbstract

class SteppingStonesSimulator(Simulator):
    """
    Simulator with stepping stones environment.
    """
    
    # Maximum average distance between feet and stones to accept a randomly
    # selected set of 4 contact locataions 
    MAX_DIST_RANDOM_LOCATION = 0.07 # m
    # Height offset when initialising start position
    HEIGHT_OFFSET_START = -0.008 # m
    # Minimun number of steps the goal is reached consecutively
    MIN_GOAL_CONSECUTIVE = 15
    # Check if robot reached goal every <CHECK_GOAL_PERIOD> steps
    CHECK_GOAL_PERIOD = 25
    # Stone color
    DEFAULT_STONE_RGBA = [0.5, 0.5, 0.55, 1.]  # [R, G, B, A]
    STONE_CONTACT_RGBA = [1., 0.1, 0., 1.]
    STONE_CONTACT_2_RGBA = [1., 0.6, 0., 1.]
    STONE_GOAL_RGBA = [0.4, 1., 0.4, 1.]
    # Min steps in contact to consider the robot landed
    MIN_STEP_IN_CONTACT = 165 # ms
    # Noise level contact plan
    NOISE_CONTACT_POS = 2.0e-3

    
    def __init__(self,
                 stepping_stones_env: SteppingStonesEnv,
                 robot: SoloRobotWrapper,
                 controller: BiConMPC,
                 data_recorder: DataRecorderAbstract = None,
                 **kwargs) -> None:
        super().__init__(robot, controller, data_recorder)
        
        self.stepping_stones = stepping_stones_env
        self.feet_pos_0 = self.robot.get_foot_locations_world()
        self.q0, self.v0 = self.robot.get_state()
        self.start_pos = []
        self.start_indices = []
        self.goal_pos = []
        self.goal_indices = []
        self.contact_plan_id = []
        self.eeff_last_contact = []
        self.consec_on_goal = 0
        self.waiting_for_next_jump = False
        self.consecutive_landing = 0
        self.last_replanning = 0
        self.controller_copy = copy.copy(controller)

        optionals = {
            "min_goal_consecutive" : SteppingStonesSimulator.MIN_GOAL_CONSECUTIVE,            
            "height_offset" : SteppingStonesSimulator.HEIGHT_OFFSET_START,            
        }
        optionals.update(kwargs)
        for k, v in optionals.items(): setattr(self, k, v)
        
        self.shape = pybullet.GEOM_CYLINDER
        if self.stepping_stones.shape == "box":
            self.shape = pybullet.GEOM_BOX
        
        self.env2pybullet_id = {} # {id in env : id in pybullet}
        self.pybullet2env_id = {} # {id in pybullet : id in env}

        # Remove stones
        id_to_keep = self.get_closest_foot_stone_id()[0].tolist()
        self.stepping_stones.remove_random(keep = id_to_keep)
        
        self._create_stepping_stones()
    
    def _reset(self) -> None:
        self.waiting_for_next_jump = False
        self.consecutive_landing = 0
        self.last_replanning = 0
        self.eeff_last_contact = []
        return super()._reset()
            
    def _create_stepping_stones(self):
        """ 
        Create stepping stones in pybullet.
        """        
        for i_env, (loc, size) in enumerate(zip(
            self.stepping_stones.positions,
            self.stepping_stones.size)):

            h = loc[2]
            if h <= 0:
                continue
            
            r = size / 2.

            if loc[2] > 0:
                cuid = self.robot.env._p.createCollisionShape(
                    pybullet.GEOM_CYLINDER,
                    radius = r,
                    height = h)
                vuid = self.robot.env._p.createVisualShape(
                    pybullet.GEOM_CYLINDER,
                    radius = r,
                    length = h,
                    rgbaColor = SteppingStonesSimulator.DEFAULT_STONE_RGBA)
                i_pb = self.robot.env._p.createMultiBody(
                    0.,
                    cuid,
                    vuid,
                    basePosition=[loc[0] , loc[1], loc[2]/2.],
                    baseOrientation=self.robot.env._p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
                
                self.env2pybullet_id[i_env] = i_pb
                self.pybullet2env_id[i_pb] = i_env
                
    def move_stepping_stones(self, indices, new_positions):
            """
            Move the specified stepping stones to new positions.

            Args:
                indices (np.ndarray): Array of indices of the stepping stones to move.
                new_positions (np.ndarray): Array of new positions for the stepping stones.
            """
            for idx, new_pos in zip(indices, new_positions):
                if idx in self.env2pybullet_id:
                    pb_id = self.env2pybullet_id.pop(idx)
                    self.pybullet2env_id.pop(pb_id)
                    # Remove the old stone
                    self.robot.env._p.removeBody(pb_id)
                    
                    r = self.stepping_stones.size[idx] / 2.
                    h = new_pos[2]
                                        
                    # Create a new stone with updated position and size
                    cuid = self.robot.env._p.createCollisionShape(
                        pybullet.GEOM_CYLINDER,
                        radius=r,
                        height=h,
                        physicsClientId=self.robot.env.physicsClientId)
                    
                    vuid = self.robot.env._p.createVisualShape(
                        pybullet.GEOM_CYLINDER,
                        radius=r,
                        length=h,
                        rgbaColor=SteppingStonesSimulator.DEFAULT_STONE_RGBA,
                        physicsClientId=self.robot.env.physicsClientId)
                    
                    new_pb_id = self.robot.env._p.createMultiBody(
                        0.,
                        cuid,
                        vuid,
                        basePosition=[new_pos[0], new_pos[1], new_pos[2] / 2.],
                        baseOrientation=self.robot.env._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                        physicsClientId=self.robot.env.physicsClientId)

                    # Update the mappings
                    self.env2pybullet_id[idx] = new_pb_id
                    self.pybullet2env_id[new_pb_id] = idx
                    

    def remove_stepping_stones(self, indices):
        """ 
        Remove stepping stones in pybullet
        """
        
        for idx in indices:
            if idx in self.env2pybullet_id:
                pb_id = self.env2pybullet_id.pop(idx)
                self.pybullet2env_id.pop(pb_id)

                # Remove the old stone
                self.robot.env._p.removeBody(pb_id)
        
    def _sample_random_feet_stones_locations(self,
                                             max_dist_to_center: float = -1.,
                                             min_dist_to_center: float= 0.) -> list[int]:
        """
        Returns possible stones positions for the feet
        based on the nominal configuration.
        
        Args:
            max_dist_to_center (float): Maximum distance to center of the env.

        Returns:
            list[int]: indices of the 4 selected stepping stones (list)
        """
        i = 0
        position_found = False
        closest_stones_id = []
        
        while not position_found and i <= 10000:

            # Draw random center stone id
            if max_dist_to_center < 0.:
                id_stone_center = np.random.randint(self.stepping_stones.N)
            # Draw at random among the possible ones
            else:
                d = np.random.uniform(min_dist_to_center, max_dist_to_center)
                theta = np.random.uniform(-np.pi, np.pi)
                random_pos = np.array([[np.cos(theta), np.sin(theta), 1.]]) * d
                random_pos[:, 2] = self.stepping_stones.height
                
                id_stone_center, _ = self.stepping_stones.get_closest(random_pos)
            stone_center_pos = self.stepping_stones.positions[id_stone_center]

            # Check if feet in nominal position have stones closeby
            feet_pos_0 = self.robot.get_foot_locations_world()
            feet_pos_0_centered = feet_pos_0 - np.mean(feet_pos_0, axis=0, keepdims=True)
            feet_pos = stone_center_pos + feet_pos_0_centered
            closest_stones_id, closest_stones_pos = self.stepping_stones.get_closest(feet_pos)
            max_distance = np.max(np.linalg.norm(closest_stones_pos- feet_pos, axis=-1))

            if (max_distance < SteppingStonesSimulator.MAX_DIST_RANDOM_LOCATION and
                (closest_stones_pos[:, -1] > 0).all() # Not among the removed locations
                ):
                position_found = True
                
            i += 1
        
        return closest_stones_id
    
    def reset_stones_environment(self):
        self.stepping_stones.init_stones()
        self.remove_stepping_stones(np.arange(self.stepping_stones.N))
        id_to_keep = self.start_pos if len(self.start_pos) > 0 else self.get_closest_foot_stone_id()[0]
        self.stepping_stones.remove_random(keep = id_to_keep)
        self._create_stepping_stones()
    
    def get_closest_foot_stone_id(self) -> np.ndarray:
        foot_ini_pos_world = self.robot.get_foot_locations_world()
        start_indices = self.stepping_stones.get_closest(foot_ini_pos_world)[0]
        return start_indices, foot_ini_pos_world
            
    def set_start_and_goal(self,
                           start_indices: np.ndarray | list[int] | Any = [],
                           goal_indices: np.ndarray | list[int] | Any = [],
                           max_dist: float = .42,
                           min_dist: float = .28,
                           init_robot_pos: bool = True,
                           randomize_state: bool = False) -> None:
        """
        Set the start and goal locations.
        If not provided, set them at random so that the start is at the center
        and the goal at a maximum distance from the center.
        """
        # Start stones
        if ((isinstance(start_indices, list) or isinstance(start_indices, np.ndarray))
            and
            (len(start_indices) == 4 and not init_robot_pos)):
            self.start_indices = start_indices
            start_pos_world = self.stepping_stones.positions[self.start_indices]
        else:
            self.start_indices, start_pos_world = self.get_closest_foot_stone_id()
                
        # Goal stones
        if ((isinstance(goal_indices, list) or
             isinstance(goal_indices, np.ndarray)) and
            len(goal_indices) == 4):
            self.goal_indices = goal_indices
        else:
            self.goal_indices = self._sample_random_feet_stones_locations(max_dist, min_dist)

        # Set robot start state
        if init_robot_pos:
            
            # Init positions
            if randomize_state:
                q0, v0 = self.robot.reset_state_with_rand()
            else:
                q0, v0 = self.robot.reset_state()
            q_start = copy.deepcopy(q0)
            q_start[2] += self.stepping_stones.height + SteppingStonesSimulator.HEIGHT_OFFSET_START
            self.robot.reset_state(q_start, v0)

            # Move start positions under the feet
            new_stone_pos_world = self.stepping_stones.set_start_position(start_pos_world)
            self.move_stepping_stones(self.start_indices, new_stone_pos_world)
            
            # Set goal stones to the same height
            # self.stepping_stones.positions[self.goal_indices, -1] = self.stepping_stones.height
            # self.move_stepping_stones(self.goal_indices, self.stepping_stones.positions[self.goal_indices])
        
        # Change stone color
        self.reset_stones_color()
        self.color_stones(self.start_indices, np.array([SteppingStonesSimulator.STONE_CONTACT_RGBA]*4))
        self.color_stones(self.goal_indices, np.array([SteppingStonesSimulator.STONE_GOAL_RGBA]*4))
        

    def _on_goal(self) -> bool:
        """
        Checks if the robot is on the goal locations.

        Returns:
            bool: True if on goal.
        """
        # {eeff_name : id(static_geom_name)}
        eeff_contact_with = []
        on_goal = False
        
        # Filter contacts
        # Get all contact points
        contact_points = self.robot.env._p.getContactPoints(self.robot.robot_id)

        for cnt in contact_points:
            if (cnt[1] in self.pybullet2env_id.keys() and
                cnt[4] in self.robot.bullet_endeff_ids):
                stone_id = self.pybullet2env_id[cnt[1]]

            elif (cnt[2] in self.pybullet2env_id.keys() and
                  cnt[3] in self.robot.bullet_endeff_ids):
                stone_id = self.pybullet2env_id[cnt[2]]

            else:
                continue
            
            eeff_contact_with.append(stone_id)
            self.eeff_last_contact.append(stone_id)

        # Check all feet in contact are on goal location
        if (set(self.goal_indices) == set(eeff_contact_with) or
            set(self.goal_indices) == set(self.eeff_last_contact[-8:])):
            on_goal = True
            self.consec_on_goal += 1
            self.eeff_last_contact = []
            
        elif len(eeff_contact_with) > 0:
            self.consec_on_goal = 0

        return on_goal
    
    def _stop_sim(self) -> bool:
        
        if super()._stop_sim():
            return True
        
        elif self.sim_step % 5 == 0:
                feet_pos_w = self.robot.get_foot_locations_world()
                feet_height_w = feet_pos_w[:, -1]
                
                if (feet_height_w < self.stepping_stones.height / 3.).any():
                    if self.verbose: print("robot leg too low")
                    return True
        return False
        
    def _update_consecutive_landing(self) -> bool:
        eeff_contact = self.robot.get_current_eeff_contacts()
        N_contact = sum(eeff_contact)
        landed = N_contact == 4
        # Robot jumping or in the air
        if not landed:
            self.consecutive_landing = 0
            self.waiting_for_next_jump = True
        # Robot making contact or staying in contact
        else:
            self.consecutive_landing += 1

        return landed
    
    def _record_data(self) -> None:
        
        # self._update_consecutive_landing()
        # # Record data only when the robot has landed
        # if (self.consecutive_landing > SteppingStonesSimulator.MIN_STEP_IN_CONTACT and
        #     self.waiting_for_next_jump):
            
        #     self.data_recorder.record(self.q, self.v, contact_plan_id=self.contact_plan_id)
        #     self.waiting_for_next_jump = False
        if (self.controller.replan_contact() and
            self.controller.replanning != self.last_replanning and
            self.controller.replanning != 0):
            self.data_recorder.record(self.q, self.v, contact_plan_id=self.contact_plan_id)
            self.last_replanning = self.controller.replanning
    
    def _simulation_step(self) -> None:
        super()._simulation_step()
            
        if self.sim_step % SteppingStonesSimulator.CHECK_GOAL_PERIOD == 0:
            self._on_goal()
            if self.consec_on_goal >= self.min_goal_consecutive:
                self.stop_sim = True
                
    def color_stones(self, stone_indices : np.ndarray, new_color : np.ndarray):
        """
        Color stepping stones.
        """
        for id_env, color in zip(stone_indices, new_color):
            if id_env in self.env2pybullet_id.keys():
                id_pybullet = self.env2pybullet_id[id_env]
                visual_data = self.robot.env._p.getVisualShapeData(id_pybullet, -1)
                old_color = list(visual_data[0][7])
                if visual_data and (old_color != color).any():
                    self.robot.env._p.changeVisualShape(id_pybullet, -1, rgbaColor=color)
    
    def reset_stones_color(self):
        for id_pybullet in list(self.env2pybullet_id.values()):
            visual_data = self.robot.env._p.getVisualShapeData(id_pybullet, -1)
            old_color = list(visual_data[0][7])
            if visual_data and old_color != SteppingStonesSimulator.DEFAULT_STONE_RGBA:
                self.robot.env._p.changeVisualShape(id_pybullet, -1, rgbaColor=SteppingStonesSimulator.DEFAULT_STONE_RGBA)
        
    def run_contact_plan(self,
                         contact_plan_id: np.ndarray,
                         use_viewer: bool = False,
                         visual_callback_fn: Callable = None,
                         **kwargs,
                         ) -> int:
        """
        Run simulation and controller with a given contact plan.

        Args:
            - contact_plan_id (np.ndarray): Indices of the contact locations. Shape [L, Neeff, 1].
            - use_viewer (bool, optional): Use viewer. Defaults to False.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
            that create visual geometries using the mjv_initGeom function.
            See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
            for an example.
        Returns:
            int: 1 if goal reached else 0.
        """
        randomize = kwargs.get("randomize", False)
        
        self.robot.reset_state()

        # Update start and goal
        self.set_start_and_goal(contact_plan_id[0], contact_plan_id[-1], randomize_state=randomize)

        # Set goal for learned and raibert controller
        try:
            self.controller.set_goal(contact_plan_id[-1])
        except:
            print("Cannot set goal to the controller.")
            
        # Set contact plan
        self.contact_plan_id = contact_plan_id
        contact_plan_pos = self.stepping_stones.positions[self.contact_plan_id]
        
        # Randomize contact plan
        if randomize:
            contact_plan_pos[:, :, :2] += np.random.randn(*contact_plan_pos[:, :, :2].shape) * 8.e-3
        
        self.controller.set_contact_plan(contact_plan_pos)

        # Run sim
        max_sim_time = len(self.contact_plan_id) * 3
        super().run(use_viewer=use_viewer,
                    visual_callback_fn=visual_callback_fn,
                    stop_on_collision=True,
                    simulation_time=max_sim_time,
                    **kwargs
                    )
        
        goal_reached = (
            not(self.robot.collided) and
            self.consec_on_goal >= SteppingStonesSimulator.MIN_GOAL_CONSECUTIVE
            )

        # Save if goal reached
        if goal_reached:
            self.data_recorder.save() # Skip first two jumps
        else:
            self.data_recorder.reset()
            
        # Reset flags
        self.consec_on_goal = 0 
        self.waiting_for_next_jump = False
        self.consecutive_landing = 0
        
        return (int(goal_reached) * 2) - 1
    
    def reach_goal( self,
                    goal_indices: np.ndarray,
                    use_viewer: bool = False,
                    visual_callback_fn: Callable = None,
                    **kwargs,
                    ) -> int:

        self.contact_plan_id = [self.goal_indices]
        self.robot.reset_state()
        # Update goal
        self.set_start_and_goal(goal_indices=goal_indices) # Will be set as random if not provided

        # Set goal plan
        try:
            self.controller.set_goal(self.goal_indices)
        except:
            print("Cannot set goal to the controller.")

        # Run sim
        super().run(use_viewer=use_viewer,
                    visual_callback_fn=visual_callback_fn,
                    stop_on_collision=True,
                    **kwargs
                    )
        
        goal_reached = (
            not(self.robot.collided) and
            self.consec_on_goal >= SteppingStonesSimulator.MIN_GOAL_CONSECUTIVE
            )

        # Save if goal reached
        if goal_reached:
            self.data_recorder.save() # Skip first two jumps
        else:
            self.data_recorder.reset()
            
        # Reset flags
        self.consec_on_goal = 0
                
        return (int(goal_reached) * 2) - 1
    
            
if __name__ == "__main__":
    
    from py_pin_wrapper.abstract.robot import SoloRobotWrapper
    from mpc_controller.bicon_mpc import BiConMPC
    from mpc_controller.motions.cyclic.solo12_trot import trot
    from mpc_controller.motions.cyclic.solo12_jump import jump
    from utils.rendering import desired_contact_locations_callback
    from tree_search.data_recorder import JumpDataRecorder

    
    stepping_stones_height = 0.1
    stones_env = SteppingStonesEnv(
        height=stepping_stones_height,
        randomize_height_ratio=0.3,
        randomize_pos_ratio=0.,
        size_ratio=(0.65, 0.65),
        N_to_remove=0
        )
    
    robot = SoloRobotWrapper()

    controller = BiConMPC(robot, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)
    
    data_recorder = JumpDataRecorder(robot, stones_env, "test")
    
    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot,
        controller=controller,
        data_recorder=data_recorder,
        )
    
    start_indices = [51, 33, 49, 31]
    contact_plan_id = np.array([start_indices] * 20)
    contact_plan_id = np.array([[51, 33, 49, 31],  [51, 33, 49, 31], [51, 33, 49, 31], [52, 34, 50, 32], [52, 34, 50, 32], [53, 35, 51, 33]])
        
    contact_plan_callback = lambda env, sim_step, q, v : desired_contact_locations_callback(env, sim_step, q, v, controller)
    success = sim.run_contact_plan(contact_plan_id, use_viewer=True, visual_callback_fn=contact_plan_callback, verbose=True)
    print("Success:", success)