# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

import copy
from typing import Any
import numpy as np
import time

from mpc_controller.cyclic_gait_gen import CyclicQuadrupedGaitGen
from mpc_controller.robot_id_controller import InverseDynamicsController
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract


class BiConMPC(ControllerAbstract):
    REPLANNING_TIME = 0.05 # replanning time, s
    SIM_OPT_LAG = False # Take optimization time delay into account
    HEIGHT_OFFSET = 0. # Offset the height of the contact plan
    DEFAULT_SIM_DT = 1.0e-3 # s

    def __init__(self,
                 robot: PinQuadRobotWrapper,
                 **kwargs
                 ) -> None:
        super().__init__(robot, **kwargs)
        
        self.robot = robot
        
        # Optional arguments
        self.optionals = {
            "replanning_time" : BiConMPC.REPLANNING_TIME,
            "sim_opt_lag" : BiConMPC.SIM_OPT_LAG,
            "height_offset" : BiConMPC.HEIGHT_OFFSET,
            "sim_dt" : BiConMPC.DEFAULT_SIM_DT,
        }
        self.optionals.update(**kwargs)
        for k, v in self.optionals.items(): setattr(self, k, v)
        
        # Gait generator
        self.gait_gen = None
        # Gait parameters
        self.gait_params = None
        # Desired linear velocity (x, y, z)
        self.v_des = None
        # Desired angular velocity (x, y, z)
        self.w_des = None
        # Desired contact plan [H, Neef, 3]
        self.contact_plan_des = []
        self.full_length_contact_plan = []
        self.mpc_cnt_plan_w = []
        self.replanning = 0 # Replan contacts
        # True if MPC diverges
        self.diverged = False

        # Inverse dynamics controller
        self.robot_id_ctrl = InverseDynamicsController(
            robot.pin_robot,
            eff_arr=[self.robot.frame_name2id[name] for name in self.robot.foot_names])

        # MPC timings parameters
        self.sim_t = 0.0
        self.index = 0
        self.step = 0
        self.pln_ctr = 0
        self.horizon = int(self.replanning_time / self.sim_dt) # s
        self.gait_horizon = 0
        self.gait_period = 0.
       
        # Init plans
        self.xs_plan = np.empty((self.horizon, self.robot.nq + self.robot.nv), dtype=np.float32)
        self.us_plan = np.empty((self.horizon, self.robot.nv), dtype=np.float32)
        self.f_plan = np.empty((self.horizon, self.robot.ne*3), dtype=np.float32)
        
        self.set_command()
        
    @staticmethod
    def reinitialize_controller(controller: 'BiConMPC'):
        """
        Reinitialize the BiConMPC controller.
        
        Args:
            controller (BiConMPC): The BiConMPC controller instance to reinitialize.
        """
        # Store the current configuration and parameters
        kwargs = {k : v for k, v in controller.optionals.items()}
        controller.robot.reset()
        robot = copy.copy(controller.robot)
        gait_params = copy.deepcopy(controller.gait_params)

        # Reinitialize the controller
        controller.__init__(robot, **kwargs)
        
        # Set the gait parameters again
        controller.set_gait_params(gait_params)
    
    def reset(self):
        """
        Reset controller.
        """
        self.reinitialize_controller(self)
        
    # def reset(self):
    #     """
    #     Reset controller.
    #     """
    #     gait_params = copy.deepcopy(self.gait_params)
    #     self.robot.reset()
    #     robot = copy.copy(self.robot)
    #     kwargs = copy.deepcopy(self.optionals)
    #     self.__init__(robot, **kwargs)
    #     self.set_gait_params(gait_params)
        
    def replan_contact(self) -> bool:
        '''
        True if contact has to be replanned
        '''
        m = int(self.gait_period / self.replanning_time)
        return self.replanning % m == 0
        
    def set_command(self,
                    v_des: np.ndarray = np.zeros((3,)),
                    w_des: float = 0.,
                    ) -> None:
        """
        Set velocities command in world frame.

        Args:
            v_des (np.array, optional): Linear velocities (x, y, z). Defaults to np.zeros((3,)).
            w_des (float, optional): Angular velocities (x, y, z). Defaults to 0..
        """
        self.v_des = v_des
        self.w_des = w_des
        
    def set_contact_plan(self,
                         contact_plan_des: np.ndarray,
                         timings_between_switch: float = 0.,
                         ) -> None:
        """
        Set custom contact plan for the defined gait.
        Contact plan is expressed in world frame.
        No custom timings.

        Args:
            - contact_plan_des (np.array): Contact plan of shape [L, Neeff, 3].
            with L, the length of the contact plan, Neeff, the number of end effector.
            - timings_between_switch (float): Duration between two set of contacts in s.
        """
        assert self.gait_horizon > 0, "Set the gait parameters first."
        self.reset()
        # TODO: Implement timings_between_switch
        self.contact_plan_des = contact_plan_des
        # Expend the contact plan, shape [H * L, 4, 3]
        self.full_length_contact_plan = np.repeat(contact_plan_des, self.gait_horizon, axis=0)
    
    def set_gait_params(self,
                        gait_params:BiconvexMotionParams,
                       ) -> None:
        """
        Set gait parameters of the gait generator.

        Args:
            gait_params (BiconvexMotionParams): Custom gait parameters. See BiconvexMotionParams.
        """
        self.gait_params = gait_params
        self.gait_gen = CyclicQuadrupedGaitGen(self.robot, self.gait_params, self.replanning_time, self.height_offset, self.sim_dt)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        self.gait_horizon = self.gait_gen.horizon
        self.gait_period = self.gait_gen.params.gait_period

    def _step(self) -> None:
        self.pln_ctr = int((self.pln_ctr + 1)%(self.horizon))
        self.index += 1
        self.step += 1
        
    def _contains_nan(self, plan) -> bool:
        """
        Check if plan contains nan values.
        """
        return np.isnan(plan).any()
        
    def get_desired_contacts(self, q, v) -> np.ndarray:
        """
        Returns the desired contact positions for the <horizon>
        next timesteps of the MPC based on the desired contact plan.
        Should be called before the MPC is called.

        Returns:
            np.ndarray: Contact plan. Shape [H, 4, 3].
        """
        
        mpc_contacts = []
        if len(self.contact_plan_des) > 0:
            
            # Stay on the last contact location if end of contact plan is reached
            if self.replanning + 2 * self.gait_horizon > len(self.full_length_contact_plan):
                self.full_length_contact_plan = np.concatenate(
                        (
                        self.full_length_contact_plan,
                        np.repeat(self.full_length_contact_plan[-1, np.newaxis, :, :], 2 * self.gait_horizon,
                        axis=0
                        )),
                    axis=0
                )

            # Take the next <horizon> contact locations
            mpc_contacts = self.full_length_contact_plan[self.replanning: self.replanning + self.gait_horizon]
            # Update the desired velocity
            i_next_jump = self.replanning + 2 * (self.gait_horizon - 2)
            center_position_next_cnt = np.mean(self.full_length_contact_plan[i_next_jump], axis=0)
            self.v_des = np.round((center_position_next_cnt - q[:3]) / self.gait_period, 2)
            # Scale velocity
            self.v_des *= np.array([1.3, 2., 0.])

        self.replanning += 1
        return mpc_contacts
            
    def get_torques(self,
                    q: np.ndarray,
                    v: np.ndarray,
                    robot_data: Any
                    ) -> dict[float]:
        """
        Returns torques from simulation data.

        Args:
            q (np.array): position state (nq)
            v (np.array): velocity state (nv)
            robot_data (MjData): MuJoco simulation robot data

        Returns:
            dict[float]: torque command {joint_name : torque value}
        """
        
        sim_t = round(robot_data.time, 3)
        
        # Replanning
        if self.pln_ctr == 0:
            pr_st = time.time()

            # Contact plan in world frame
            self.mpc_cnt_plan_w = self.get_desired_contacts(q, v)
            self.xs_plan, self.us_plan, self.f_plan = self.gait_gen.optimize(
                q,
                v,
                sim_t,
                self.v_des,
                self.w_des,
                cnt_plan_des=self.mpc_cnt_plan_w)
            
            self.diverged = (self._contains_nan(self.xs_plan) or
                             self._contains_nan(self.us_plan) or
                             self._contains_nan(self.f_plan))
            
            pr_et = time.time() - pr_st
            self.index = 0
        
        # Second loop onwards lag is taken into account
        if (self.step > 0 and
            self.sim_opt_lag and
            self.step >= int(self.replanning_time/self.sim_dt)
            ):
            lag = int((1/self.sim_dt)*(pr_et - pr_st))
            self.index = lag
            
        # If no lag (self.lag < 0)
        elif (not self.sim_opt_lag and
              self.pln_ctr == 0 and
              self.step >= int(self.replanning_time/self.sim_dt)
              ):
            self.index = 0

        # Compute torques
        tau = self.robot_id_ctrl.id_joint_torques(
            q,
            v,
            self.xs_plan[self.index][:self.robot.nq],
            self.xs_plan[self.index][self.robot.nq:],
            self.us_plan[self.index],
            self.f_plan[self.index])

        # Create command {joint_name : torque value}
        torque_command = {
            joint_name: tau[id]
            for joint_name, id
            in self.robot.joint_name2act_id.items()
        }

        # Increment timing variables
        self._step()
        
        return torque_command