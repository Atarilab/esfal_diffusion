import os
from typing import Any, Callable
import pybullet
import time
import numpy as np

from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from py_pin_wrapper.abstract.controller import ControllerAbstract
from py_pin_wrapper.abstract.data_recorder import DataRecorderAbstract

from robot_properties_solo.config import Solo12Config

class Simulator(object):
    DEFAULT_VIDEO_DIR = "/home/atari_ws/project/figures/video/"
    def __init__(self,
                 robot: SoloRobotWrapper,
                 controller: ControllerAbstract,
                 data_recorder: DataRecorderAbstract = None,
                 ) -> None:
        
        self.robot = robot
        self.controller = controller
        self.data_recorder = (data_recorder
                              if data_recorder != None
                              else DataRecorderAbstract()
                              )
        self.sim_dt = Solo12Config.control_period
        self.sim_time = 0.
        self.sim_step = 0
        self.simulation_it_time = []
        self.q, self.v = None, None
        self.visual_callback_fn = None
        self.verbose = False
        self.stop_sim = False  # Eventually for inherited class

    def _reset(self) -> None:
        """
        Reset flags and timings.
        """
        self.sim_time = 0.
        self.sim_step = 0
        self.simulation_it_time = []
        self.verbose = False
        self.stop_sim = False
        self.robot.collided = False
        self.controller.reset()
        
    def _record_data(self) -> None:
        """
        Record data with data recorder.
        To be overridden.
        """
        self.data_recorder.record(self.q, self.v, time=self.sim_time)
        
    def _simulation_step(self) -> None:
        """
        Main simulation step.
        - Record data
        - Compute and apply torques
        - Step simulation
        """
        # Get state in Pinocchio format (x, y, z, qx, qy, qz, qw)
        self.q, self.v = self.robot.get_state()
        
        # Record data
        self._record_data()
        
        # Torques should be a map {joint_name : torque value}
        torques = self.controller.get_torques(self.q, self.v)
        # Apply torques
        self.robot.send_joint_command(torques)

        # pybullet sim step
        self.robot.step_simulation()
        self.sim_time += self.sim_dt
        self.sim_step += 1
        
        # TODO: Add external disturbances
        
    def _simulation_step_with_timings(self,
                                      real_time: bool,
                                      ) -> None:
        """
        Simulation step with time keeping and timings measurements.
        """
        step_start = time.time()
        self._simulation_step()
        step_duration = time.time() - step_start
        
        self.simulation_it_time.append(step_duration)
        
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.sim_dt - step_duration
        if real_time and time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
    def _stop_sim(self) -> bool:
        """
        True if the simulation has to be stopped.

        Returns:
            bool: stop simulation
        """        
        if self.stop_on_collision and (self.robot.collided or self.robot.is_collision()):
            if self.verbose: print("/!\ Robot collision")
            return True

        if self.stop_sim:
            if self.verbose: print("/!\ Simulation stopped")
            return True
        
        if self.controller.diverged:
            if self.verbose: print("/!\ Controller diverged")
            return True
        
        return False
    
    def _get_video_path(self, video_save_dir : str = "") -> str:
        """
        Return full video file path.
        """
        date_time_str = time.strftime('%d%m-%H%M')
        file_name = f"run_{date_time_str}.mp4"
        if video_save_dir == "":  
            video_save_dir = Simulator.DEFAULT_VIDEO_DIR
        os.makedirs(video_save_dir, exist_ok=True)
        
        video_save_path = os.path.join(video_save_dir, file_name)
        
        return video_save_path
    
    def run(self,
            simulation_time: float = -1.,
            use_viewer: bool = True,
            visual_callback_fn: Callable = None,
            **kwargs,
            ) -> None:
        """
        Run simulation for <simulation_time> seconds with or without a viewer.

        Args:
            - simulation_time (float, optional): Simulation time in second.
            Unlimited if -1. Defaults to -1.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
            that create visual geometries using the mjv_initGeom function.
            See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
            for an example.
            - viewer (bool, optional): Use viewer. Defaults to True.
            - verbose (bool, optional): Print timing informations.
            - stop_on_collision (bool, optional): Stop the simulation when there is a collision.
        """
        real_time = kwargs.get("real_time", use_viewer)
        self.verbose = kwargs.get("verbose", False)
        self.stop_on_collision = kwargs.get("stop_on_collision", False)
        self.visual_callback_fn = visual_callback_fn
        record_video = kwargs.get("record_video", False)
        video_save_dir = kwargs.get("video_save_dir", "")
        
        ### Start video recording
        if record_video:
            video_save_path = self._get_video_path(video_save_dir)
            self.robot.env.start_video_recording(video_save_path)

        if self.verbose:
            print("-----> Simulation start")
        
        sim_start_time = time.time()
        self.sim_step = 0
        
        while (simulation_time < 0. or
                self.sim_step < simulation_time * (1 / self.sim_dt)
                ):
            self._simulation_step_with_timings(real_time)
            if use_viewer:
                self.update_visuals()
                
            if self._stop_sim():
                break
    
        if record_video:
            self.robot.env.stop_video_recording()
            
        if self.verbose:
            print(f"-----> Simulation end\n")
            sum_step_time = sum(self.simulation_it_time)
            mean_step_time = sum_step_time / len(self.simulation_it_time)
            total_sim_time = time.time() - sim_start_time
            print(f"--- Total optimization step time: {sum_step_time:.2f} s")
            print(f"--- Mean simulation step time: {mean_step_time*1000:.2f} ms")
            print(f"--- Total simulation time: {total_sim_time:.2f} s")

        # Reset flags
        self._reset()

    def update_visuals(self) -> None:
        """
        Update visuals according to visual_callback_fn.
        """
        if self.visual_callback_fn != None:
            try:
                self.visual_callback_fn(self.robot.env, self.sim_step, self.q, self.v)
                
            except Exception as e:
                if self.verbose:
                    print("Can't update visual geometries.")
                    print(e)
                    
                    
if __name__ == "__main__":
    
    from py_pin_wrapper.abstract.robot import SoloRobotWrapper
    from mpc_controller.bicon_mpc import BiConMPC
    from mpc_controller.motions.cyclic.solo12_trot import trot
    from mpc_controller.motions.cyclic.solo12_jump import jump
    from utils.rendering import desired_contact_locations_callback

    robot = SoloRobotWrapper(server=pybullet.GUI)
    controller = BiConMPC(robot)
    controller.set_gait_params(jump)
    v_des = np.array([0., 0.3, 0.])
    controller.set_command(v_des)
    
    sim = Simulator(
        robot=robot,
        controller=controller,
        )
    
    contact_plan_callback = lambda env, sim_step, q, v : desired_contact_locations_callback(env, sim_step, q, v, controller)
    sim.run(stop_on_collision=False, visual_callback_fn=contact_plan_callback)