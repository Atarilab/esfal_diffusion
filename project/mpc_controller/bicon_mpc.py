import copy
import numpy as np
import time
from robot_properties_solo.config import Solo12Config


from mpc_controller.cyclic_gait_gen import CyclicQuadrupedGaitGen
from mpc_controller.robot_id_controller import InverseDynamicsController
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from py_pin_wrapper.abstract.robot import SoloRobotWrapper
from py_pin_wrapper.abstract.controller import ControllerAbstract

class BiConMPC(ControllerAbstract):
    REPLANNING_TIME = 0.05 # replanning time, s
    SIM_OPT_LAG = False # Take optimization time delay into account
    HEIGHT_OFFSET = 0. # Offset the height of the contact plan

    def __init__(self,
                 robot: SoloRobotWrapper,
                 **kwargs
                 ) -> None:
        super(BiConMPC, self).__init__(robot, **kwargs)
        
        self.robot = robot
        self.pin_robot = self.robot.pinocchio_robot
        self.nq = self.pin_robot.nq
        self.nv = self.pin_robot.nv
        self.max_torque = 10
        
        # Optional arguments
        self.optionals = {
            "replanning_time" : BiConMPC.REPLANNING_TIME,
            "sim_opt_lag" : BiConMPC.SIM_OPT_LAG,
            "height_offset" : BiConMPC.HEIGHT_OFFSET,
        }
        self.optionals.update(**kwargs)
        for k, v in self.optionals.items(): setattr(self, k, v)
        
        # Gait generator
        self.gait_gen = None
        # Desired linear velocity (x, y, z)
        self.v_des = None
        # Desired angular velocity (x, y, z)
        self.w_des = None
        # Desired contact plan [H, Neef, 3]
        self.contact_plan_des = []
        self.full_length_contact_plan = []
        self.replanning = 0 # Replan contacts
        # True if MPC diverges
        self.diverged = False

        # Inverse dynamics controller
        self.robot_id_ctrl = InverseDynamicsController(
            self.pin_robot,
            self.robot.config.end_effector_names)

        # MPC timings parameters
        self.sim_t = 0.0
        self.sim_dt = self.robot.config.control_period
        self.index = 0
        self.step = 0
        self.pln_ctr = 0
        self.horizon = int(self.replanning_time / self.sim_dt) # s
        self.gait_horizon = 0
        self.gait_period = 0.
       
        # Init plans
        self.xs_plan = np.empty((3*self.horizon, self.nq + self.nv), dtype=np.float32)
        self.us_plan = np.empty((3*self.horizon, self.nv), dtype=np.float32)
        self.f_plan = np.empty((3*self.horizon, self.robot.config.nb_joints), dtype=np.float32)
        
        self.set_command()
        
    @staticmethod
    def reinitialize_controller(controller: 'BiConMPC'):
        """
        Reinitialize the BiConMPC controller.
        
        Args:
            controller (BiConMPC): The BiConMPC controller instance to reinitialize.
        """
        # Store the current configuration and parameters
        kwargs = {k: v for k, v in controller.optionals.items()}
        robot = controller.robot
        gait_params = controller.gait_params

        # Reinitialize the controller
        controller.__init__(robot, **kwargs)
        
        # Set the gait parameters again
        controller.set_gait_params(gait_params)
    
    def reset(self):
        """
        Reset controller.
        """
        self.reinitialize_controller(self)
        
    def replan_contact(self) -> bool:
        '''
        True if contact has to be replanned
        '''
        return int(self.replanning % (self.gait_period // self.replanning_time)) == 0
        
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
        self.gait_gen = CyclicQuadrupedGaitGen(self.robot, self.gait_params, self.replanning_time, self.height_offset)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)
        self.gait_horizon = self.gait_gen.horizon
        self.gait_period = self.gait_gen.params.gait_period

    def _step(self) -> None:
        self.sim_t += self.sim_dt
        self.pln_ctr = int((self.pln_ctr + 1)%(self.horizon))
        self.index += 1
        self.step += 1
        
    def _check_if_diverged(self, plan):
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
            if self.replanning + self.gait_horizon * 2 > len(self.full_length_contact_plan):
                self.full_length_contact_plan = np.concatenate(
                        (
                        self.full_length_contact_plan,
                        np.repeat(self.full_length_contact_plan[-1, np.newaxis, :, :], self.gait_horizon * 2,
                        axis=0
                        )),
                    axis=0
                )

            # Take the next horizon contact locations
            mpc_contacts = self.full_length_contact_plan[self.replanning:self.replanning + self.gait_horizon * 2]
            # Update the desired velocity
            i = self.gait_horizon
            avg_position_next_cnt = np.mean(mpc_contacts[i], axis=0)
            # avg_position_cnt = np.mean(mpc_contacts[0], axis=0)
            self.v_des = np.round((avg_position_next_cnt - q[:3]) / self.gait_period, 2)
            self.v_des *= 1.45
            self.v_des[-1] = 0.

        return mpc_contacts
            
    def get_torques(self,
                    q: np.ndarray,
                    v: np.ndarray,
                    **kwargs
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

        # Replanning
        if self.pln_ctr == 0:
            pr_st = time.time()
            
            self.xs_plan, self.us_plan, self.f_plan = self.gait_gen.optimize(
                q,
                v,
                self.sim_t,
                self.v_des,
                self.w_des,
                cnt_plan_des=self.get_desired_contacts(q, v))
            self.replanning += 1
            
            self.diverged = (self._check_if_diverged(self.xs_plan) or
                             self._check_if_diverged(self.us_plan) or
                             self._check_if_diverged(self.f_plan))
            
            pr_et = time.time() - pr_st
        
        # Second loop onwards lag is taken into account
        if (
            self.step > 0 and
            self.sim_opt_lag and
            self.step > int(self.replanning_time/self.sim_dt) - 1
            ):
            lag = int((1/self.sim_dt)*(pr_et - pr_st))
            self.index = lag
        # If no lag (self.lag < 0)
        elif (
            not self.sim_opt_lag and
            self.pln_ctr == 0. and
            self.step > int(self.replanning_time/self.sim_dt) - 1
        ):
            self.index = 0

        # Compute torques
        tau = self.robot_id_ctrl.id_joint_torques(
            q,
            v,
            self.xs_plan[self.index][:self.nq].copy(),
            self.xs_plan[self.index][self.nq:].copy(),
            self.us_plan[self.index],
            self.f_plan[self.index],
            [])

        tau = np.clip(tau, -self.max_torque , self.max_torque)

        # Create command {joint_name : torque value}
        torque_command = {
            joint_name: torque_value
            for joint_name, torque_value
            in zip(self.robot.joint_names, tau)
        }

        # Increment timing variables
        self._step()
        
        return torque_command
