
import copy
from typing import Any, Callable, Tuple
import numpy as np
import mujoco
import threading
import keyboard
import time

from environment.stepping_stones import SteppingStonesEnv
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.bicon_mpc import BiConMPC
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract

class SteppingStonesSimulator(Simulator):
    """
    Simulator with stepping stones environment.
    """
    
    # Maximum average distance between feet and stones to accept a randomly
    # selected set of 4 contact locataions 
    MAX_DIST_RANDOM_LOCATION = 0.07 # m
    # Height offset when initialising start position
    HEIGHT_OFFSET_START = 0.02 # m
    # Minimun number of steps the goal is reached consecutively
    MIN_GOAL_CONSECUTIVE = 9
    # Check if robot reached goal every <CHECK_GOAL_PERIOD> steps
    CHECK_GOAL_PERIOD = 90
    # Stone color
    DEFAULT_STONE_RGBA = [0.5, 0.5, 0.55, 1.]  # [R, G, B, A]
    STONE_CONTACT_RGBA = [168/255, 0., 17/255, 1.]
    STONE_CONTACT_2_RGBA = [0.8, 0.6, 0., 1.]
    STONE_GOAL_RGBA = [15/255, 130/255, 35/255, 1.]
    # Min steps in contact to consider the robot landed
    MIN_STEP_IN_CONTACT = 165 # ms
    # Noise level contact plan
    NOISE_CONTACT_POS = 3.0e-3

    
    def __init__(self,
                 stepping_stones_env: SteppingStonesEnv,
                 robot: MJQuadRobotWrapper,
                 controller: BiConMPC,
                 data_recorder: DataRecorderAbstract = None,
                 goal_dist: Tuple[float, float] = (0.35, 0.5),
                 **kwargs) -> None:
        super().__init__(robot, controller, data_recorder)
        
        self.stepping_stones = stepping_stones_env
        self.feet_pos_0 = self.robot.get_foot_pos_world()
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
        self.controller = controller
        self.goal_dist = goal_dist

        optionals = {
            "min_goal_consecutive" : SteppingStonesSimulator.MIN_GOAL_CONSECUTIVE,            
            "height_offset" : SteppingStonesSimulator.HEIGHT_OFFSET_START,
        }
        optionals.update(kwargs)
        for k, v in optionals.items(): setattr(self, k, v)
        
        if self.stepping_stones.shape == "cylinder":
            self.stone_geom = mujoco.mjtGeom.mjGEOM_CYLINDER
        if self.stepping_stones.shape == "box":
            self.stone_geom = mujoco.mjtGeom.mjGEOM_BOX
            
        self.env2mujoco_id = {} # {id in env : id in mujoco}
        self.mujoco2env_id = {} # {id in mujoco : id in env}

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
        Create stepping stones in MuJoCo.
        """
        # New uncompiled spec
        spec = self.robot.get_mj_spec()

        for i_stone, (pos, size) in enumerate(zip(
                self.stepping_stones.positions,
                self.stepping_stones.size
            )):
            h = pos[2] / 2.
            if h <= 0:
                continue
            r = size / 2.
            
            geom = spec.worldbody.add_geom()
            geom.pos = [pos[0], pos[1], h]
            geom_name = "static_stone_" + str(i_stone)
            geom.name = geom_name
            geom.type = self.stone_geom
            geom.size[0] = r
            geom.size[1] = h
            geom.rgba = SteppingStonesSimulator.DEFAULT_STONE_RGBA

        self.robot.recompile_model(spec)
        
        # Map the environment and MuJoCo IDs
        geom = spec.worldbody.first_geom()
        while spec.worldbody.next_geom(geom):
            geom = spec.worldbody.next_geom(geom)
            if "static" in geom.name:
                i_stone =  int(geom.name.split("static_stone_")[-1])
                self.env2mujoco_id[i_stone] = geom.id
                self.mujoco2env_id[geom.id] = i_stone
    
    def move_stepping_stones(self, indices, new_positions):
        """
        Move the specified stepping stones to new positions.

        Args:
            indices (np.ndarray): Array of indices of the stepping stones to move.
            new_positions (np.ndarray): Array of new positions for the stepping stones.
        """
        for idx, new_pos in zip(indices, new_positions):
            if idx in self.env2mujoco_id:
                geom_id = self.env2mujoco_id[idx]
                
                # Update position and size (height and radius)
                self.robot.model.geom(geom_id).pos = [new_pos[0], new_pos[1], new_pos[2]/2.]
                self.robot.model.geom(geom_id).size[0] = self.stepping_stones.size[idx] / 2.  # radius
                self.robot.model.geom(geom_id).size[1] = new_pos[2] / 2. # height

                # Update the MuJoCo data to reflect changes
                mujoco.mj_forward(self.robot.model, self.robot.data)
                
    def _get_closest_feet_pos_from_center(self, center_w : np.ndarray) -> tuple[list[int], float]:
        """
        Returns the closest stepping stone for each foot in nominal position
        when the base in at center_position.
        Also returns the average distance of the stepping stones
        in nominal configuration.
        """
        feet_pos_0 = self.robot.get_foot_pos_world()
        feet_pos_0_centered = feet_pos_0 - np.mean(feet_pos_0, axis=0, keepdims=True)
        feet_pos = center_w + feet_pos_0_centered
        closest_stones_id, closest_stones_pos = self.stepping_stones.get_closest(feet_pos)
        dist_from_nominal = np.max(np.linalg.norm(closest_stones_pos[:, :2]-feet_pos[:, :2], axis=-1))

        return closest_stones_id, dist_from_nominal
        
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
            closest_stones_id, dist_to_nominal = self._get_closest_feet_pos_from_center(stone_center_pos)

            # feet_pos_0 = self.robot.get_foot_pos_world()
            # feet_pos_0_centered = feet_pos_0 - np.mean(feet_pos_0, axis=0, keepdims=True)
            # feet_pos = stone_center_pos + feet_pos_0_centered
            # closest_stones_id, closest_stones_pos = self.stepping_stones.get_closest(feet_pos)

            def is_removed(indices):
                for i in indices:
                    if i in self.stepping_stones.id_to_remove:
                        return True
                return False
            
            if (dist_to_nominal < SteppingStonesSimulator.MAX_DIST_RANDOM_LOCATION and
                not is_removed(closest_stones_id) # Not among the removed locations
                ):
                position_found = True
            i += 1
        
        return closest_stones_id
    
    def get_closest_foot_stone_id(self) -> np.ndarray:
        foot_ini_pos_world = self.robot.get_foot_pos_world()
        start_indices = self.stepping_stones.get_closest(foot_ini_pos_world)[0]
        return start_indices, foot_ini_pos_world
            
    def set_start_and_goal(self,
                           start_indices: np.ndarray = None,
                           goal_indices: np.ndarray = None,
                           max_dist: float = .42,
                           min_dist: float = .28,
                           init_robot_pos: bool = True,
                           randomize_state: bool = False) -> None:
        """
        Set the start and goal locations.
        If not provided, set them at random so that the start is at the center
        and the goal at a maximum distance from the center.
        """
        # Set start
        if (start_indices is not None and
            (len(start_indices) == 4 and not init_robot_pos)):
            self.start_indices = start_indices
            start_pos_world = self.stepping_stones.positions[self.start_indices]
        # Set start under feet
        else:
            # Init positions
            if randomize_state:
                self.robot.reset_randomize()
                q0, v0 = self.robot.get_state()
            else:
                self.robot.reset()
                q0, v0 = self.robot.get_state()
            q_start = copy.deepcopy(q0)
            q_start[2] += self.stepping_stones.height + SteppingStonesSimulator.HEIGHT_OFFSET_START

            self.robot.update(q_start, v0)

            self.start_indices, start_pos_world = self.get_closest_foot_stone_id()
            # Move start positions under the feet
            new_stone_pos_world = self.stepping_stones.set_start_position(start_pos_world)
            self.move_stepping_stones(self.start_indices, new_stone_pos_world)
            
        # Set goal
        if (goal_indices is not None and
            len(goal_indices) == 4):
            self.goal_indices = goal_indices
        # Set random goal if not specified
        else:
            self.goal_indices = self._sample_random_feet_stones_locations(max_dist, min_dist)

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
        # id of the geometries in contact with the feet
        eeff_contact_with = []
        on_goal = False
        
        # Get foot contact pairs
        foot_contacts = list(self.robot.foot_contacts().values())
        eeff_contact_with = [self.mujoco2env_id[i] for i in foot_contacts if i > 0]

        # Check all feet in contact are on goal location
        if (len(self.goal_indices) > 0 and
            set(self.goal_indices) == set(eeff_contact_with)):
            on_goal = True
            self.consec_on_goal += 1
            
        elif (len(eeff_contact_with) > 0 and
              not set(eeff_contact_with).issubset(set(self.goal_indices))):
            self.consec_on_goal = 0

        return on_goal
    
    def _stop_sim(self) -> bool:
        
        if super()._stop_sim():
            return True
        
        elif self.sim_step % 10 == 0:
                feet_pos_w = self.robot.get_foot_pos_world()
                feet_height_w = feet_pos_w[:, -1]
                
                if (self.stepping_stones.height > 0 and 
                    (feet_height_w < self.stepping_stones.height / 3.).any()
                    ):
                    if self.verbose: print("Robot leg too low")
                    return True
        return False
        
    def _update_consecutive_landing(self) -> bool:
        """
        Update consecutive landing counter.

        Returns:
            bool: True if the robot has its for legs in contact.
        """
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
        """
        Record data.
        """
        if (self.controller.replan_contact() and
            self.controller.replanning != self.last_replanning):
            i_jump = self.controller.replanning // self.controller.gait_horizon
            self.data_recorder.record(self.q, self.v, contact_plan_id=self.contact_plan_id, i_jump=i_jump)
            self.last_replanning = self.controller.replanning
    
    def _simulation_step(self) -> None:
        super()._simulation_step()
            
        if self.sim_step % SteppingStonesSimulator.CHECK_GOAL_PERIOD == 0:
            # Check if goal reached
            self._on_goal()
            if self.consec_on_goal >= self.min_goal_consecutive:
                if self.verbose: print("Goal reached!")
                self.stop_sim = True
                
        # Change color stepping stones
        if (not self.visual_callback_fn is None and
            self.controller.replan_contact() and
            self.controller.replanning != self.last_replanning):
            current_cnt_id = self.stepping_stones.get_closest_xy(self.controller.mpc_cnt_plan_w[0])[0]
            next_cnt_id = self.stepping_stones.get_closest_xy(self.controller.mpc_cnt_plan_w[-1])[0]
            
            self.reset_stones_color()
            self.color_stones(current_cnt_id, np.array([SteppingStonesSimulator.STONE_CONTACT_RGBA]*4))
            self.color_stones(next_cnt_id, np.array([SteppingStonesSimulator.STONE_CONTACT_2_RGBA]*4))
            
                
    def color_stones(self, stone_indices: np.ndarray, new_color: np.ndarray):
        """
        Color stepping stones in MuJoCo.

        Args:
            stone_indices (np.ndarray): Array of indices of the stepping stones to color.
            new_color (np.ndarray): Array of new colors for the stepping stones.
        """
        for id_env, color in zip(stone_indices, new_color):
            if id_env in self.env2mujoco_id:
                id_mujoco = self.env2mujoco_id[id_env]
                # Check if the color is different from the current one
                if not np.allclose(self.robot.model.geom_rgba[id_mujoco], color):
                    self.robot.model.geom_rgba[id_mujoco] = color

        mujoco.mj_forward(self.robot.model, self.robot.data)

    def reset_stones_color(self):
        """
        Reset the colors of all stepping stones to the default color.
        """
        for id_mujoco in list(self.env2mujoco_id.values()):
            # Check if the color is different from the default one
            if not np.allclose(self.robot.model.geom_rgba[id_mujoco], SteppingStonesSimulator.DEFAULT_STONE_RGBA):
                self.robot.model.geom_rgba[id_mujoco] = SteppingStonesSimulator.DEFAULT_STONE_RGBA
        
        if len(self.start_indices) > 0 and len(self.goal_indices) > 0:
            self.color_stones(self.start_indices, np.array([SteppingStonesSimulator.STONE_CONTACT_RGBA]*4))
            self.color_stones(self.goal_indices, np.array([SteppingStonesSimulator.STONE_GOAL_RGBA]*4))
            
        mujoco.mj_forward(self.robot.model, self.robot.data)
    
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
        
        # Update start and goal
        self.set_start_and_goal(
            contact_plan_id[0],
            contact_plan_id[-1],
            randomize_state=randomize,
            min_dist=self.goal_dist[0],
            max_dist=self.goal_dist[1],)

        # Set contact plan
        self.contact_plan_id = contact_plan_id
        contact_plan_pos = self.stepping_stones.positions[self.contact_plan_id]
        
        # Randomize contact plan
        if randomize:
            contact_plan_pos[:, :, :2] += np.random.randn(*contact_plan_pos[:, :, :2].shape) * SteppingStonesSimulator.NOISE_CONTACT_POS
        
        self.controller.set_contact_plan(contact_plan_pos)

        # Run sim
        max_sim_time = len(self.contact_plan_id) * 3
        super().run(use_viewer=use_viewer,
                    visual_callback_fn=visual_callback_fn,
                    stop_on_collision=True,
                    simulation_time=max_sim_time,
                    **kwargs
                    )
        
        # Goal reached if not collided and on goal
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
        
        # +1 / -1
        success = (int(goal_reached) * 2) - 1
        return success
    
    def reach_goal( self,
                    goal_indices: np.ndarray,
                    use_viewer: bool = False,
                    visual_callback_fn: Callable = None,
                    **kwargs,
                    ) -> int:

        self.contact_plan_id = [self.goal_indices]

        # Update goal
        self.set_start_and_goal(goal_indices=goal_indices) # Will be set as random if not provided

        # Set goal plan
        try:
            self.controller.set_goal(self.goal_indices)
        except Exception as e:
            print("Cannot set goal to the controller.")
            print(e)

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
    
    from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
    from mpc_controller.bicon_mpc import BiConMPC
    from mpc_controller.motions.cyclic.go2_trot import trot
    from mpc_controller.motions.cyclic.go2_jump import jump
    from utils.rendering import desired_contact_locations_callback
    from tree_search.data_recorder import ContactsDataRecorder
    from utils.config import Go2Config
    from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
    from mj_pin_wrapper.sim_env.utils import RobotModelLoader
    
    stepping_stones_height = 0.001
    stones_env = SteppingStonesEnv(
        height=stepping_stones_height,
        randomize_height_ratio=0.3,
        randomize_pos_ratio=0.,
        size_ratio=(0.65, 0.65),
        N_to_remove=0
        )

    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )

    controller = BiConMPC(robot.pin, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)
    
    data_recorder = None #JumpDataRecorder(robot, stones_env, "test")
    
    sim = SteppingStonesSimulator(
        stepping_stones_env=stones_env,
        robot=robot.mj,
        controller=controller,
        data_recorder=data_recorder,
        )
    
    start_indices = [51, 33, 49, 31]
    contact_plan_id = np.array([start_indices] * 20)
    contact_plan_id = np.array([[51, 33, 49, 31],  [51, 33, 49, 31], [51, 33, 49, 31], [52, 34, 50, 32], [52, 34, 50, 32], [53, 35, 51, 33]])
        
    contact_plan_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller)
        )
    success = sim.run_contact_plan(contact_plan_id, use_viewer=True, visual_callback_fn=contact_plan_callback, verbose=True)
    print("Success:", success)


class NavigationSteppingStonesSimulator(SteppingStonesSimulator):
    STEP_SIZE = 0.1
    def __init__(self,
                 stepping_stones_env: SteppingStonesEnv,
                 robot: MJQuadRobotWrapper,
                 controller: BiConMPC, 
                 data_recorder: DataRecorderAbstract = None,
                 goal_dist: Tuple[float] = (0.35, 0.5),
                 **kwargs) -> None:
        super().__init__(stepping_stones_env, robot, controller, data_recorder, goal_dist, **kwargs)
        
        # Base goal location
        q, _ = self.robot.get_state()
        self.center_goal_w = q[:3]
        self.goal_indices, _ = self.get_closest_foot_stone_id()
        self.set_start_and_goal(self.goal_indices, self.goal_indices, init_robot_pos=True)
        
        # Goal displacement each time the keyboard is pressed
        self.step_size = kwargs.get("step_size", NavigationSteppingStonesSimulator.STEP_SIZE)
        # Don't stop when goal is reached
        self.min_goal_consecutive = np.inf
        
        self.exit_event = threading.Event()
        self.lock = threading.Lock()
        
        try:
            keyboard.is_pressed('up')
        except:
            pass
        
    def run(self,
            visual_callback_fn: Callable[..., Any] = None,
            **kwargs) -> bool:
        
        
        # Set goal and keyboard thread
        self._start_update_thread()

        # Run sim
        super().run(simulation_time=-1,
                    use_viewer=True,
                    visual_callback_fn=visual_callback_fn,
                    stop_on_collision=True,
                    **kwargs
                    )
        
        self._stop_update_thread()
        
        success = not self.robot.collided
        return success
    
    def _update_center_goal(self):
        """
        Update base goal location based on keyboard events.
        """
        while not self.exit_event.is_set():
            # Adjust the sleep duration based on your requirements
            time.sleep(0.4)
            
            # Capture arrow key inputs
            if keyboard.is_pressed('up'):
                with self.lock:
                    self.center_goal_w[0] += self.step_size
            elif keyboard.is_pressed('down'):
                with self.lock:
                    self.center_goal_w[0] -= self.step_size
            elif keyboard.is_pressed('left'):
                with self.lock:
                    self.center_goal_w[1] += self.step_size
            elif keyboard.is_pressed('right'):
                with self.lock:
                    self.center_goal_w[1] -= self.step_size
            elif keyboard.is_pressed('esc'):
                with self.lock:
                    self.stop_sim = True

            # self.goal_location = self.contact_location[self.get_id_feet_goal()]
            self._update_feet_goals()
            
            # Set goal to controller
            try:
                self.controller.set_goal(self.goal_indices)
            except Exception as e:
                print("Cannot set goal to the controller.")
                print(e)
            
    def _update_feet_goals(self):
        """
        Update feet goal locations based on base goal location.
        """
        new_feet_goal_id, dist_to_nominal = self._get_closest_feet_pos_from_center(self.center_goal_w)

        def is_removed(indices):
            if len(self.stepping_stones.id_to_remove) == 0:
                return False
            for i in indices:
                if i in self.stepping_stones.id_to_remove:
                    return True
            return False

        # Update feet goal location if close to nominal configuration
        if (dist_to_nominal < SteppingStonesSimulator.MAX_DIST_RANDOM_LOCATION and
            not is_removed(new_feet_goal_id)):
            self.goal_indices = new_feet_goal_id
            
    def _start_update_thread(self):
        update_thread = threading.Thread(target=self._update_center_goal)
        update_thread.start()

    def _stop_update_thread(self):
        self.exit_event.set()
        
    def __del__(self):
        self._stop_update_thread()