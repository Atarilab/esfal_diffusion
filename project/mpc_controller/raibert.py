import copy
import numpy as np
import pinocchio as  pin 

from environment.stepping_stones import SteppingStonesEnv
from mpc_controller.bicon_mpc import BiConMPC
from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper

class MPC_RaiberContactPlanner(BiConMPC):
    def __init__(self,
                 robot: PinQuadRobotWrapper,
                 stepping_stones_env: SteppingStonesEnv,
                 v_max : float = .35,
                 **kwargs) -> None:
        super().__init__(robot, **kwargs)
        self.goal_locations = None
        self.stones_env = stepping_stones_env
        self.v_max = v_max
        self.i_cnt_replan = 0

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
    
    def get_desired_contacts(self, q, v) -> np.ndarray:
        """
        Returns the desired contact positions for the <horizon>
        next timesteps of the MPC based on the desired contact plan.
        Should be called before the MPC is called.

        Returns:
            np.ndarray: Contact plan. Shape [H, 4, 3].
        """
        
        if self.i_cnt_replan == 0:
            self.full_length_contact_plan = np.empty((0,4,3))

        mpc_contacts_w = []
        if self.goal_locations is not None:
            
            # Set the desired velocity. In direction of the goal. Cliped to v_max
            center_goal_location = np.mean(self.goal_locations, axis=0)
            v_base_to_goal = (center_goal_location - q[:3]) / self.gait_period * 1.3
            v_des = np.clip(v_base_to_goal, -self.v_max, self.v_max)
            v_des[-1] = 0.          
            
            # Jump inplace for the first two jumps
            if self.replanning < 20:
                v_des = np.zeros(3)

            # Update the contact location only when robot is in contact
            if self.replan_contact():
                self.i_cnt_replan += 1
                
                # Compute the raibert contact plan for that desired velocity
                q_origin = copy.deepcopy(q)
                q_origin[:2] = 0.
                cnt_plan = self.gait_gen.compute_raibert_contact_plan(q_origin, v, self.sim_t, v_des, self.w_des)
                mpc_contacts_b = cnt_plan[:, :, 1:]
                height_w = copy.deepcopy(mpc_contacts_b[:, :, -1]) # z in expressed in world frame
                mpc_contacts_b[:, :, -1] -= q[2]

                # Project the contact locations to the closest stepping stones
                # Need first to express contacts in world frame
                W_T_b = pin.XYZQUATToSE3(q[:7])
                mpc_contacts_w = self.transform_points(W_T_b, mpc_contacts_b.reshape(-1, 3)).reshape(-1, 4, 3) # world frame
                mpc_contacts_w[:, :, -1] = height_w
                
                # Next two jumps
                jump1 = self.stones_env.get_closest(mpc_contacts_w[self.gait_horizon // 2])[1].reshape(-1, 4, 3)
                jump2 = self.stones_env.get_closest(mpc_contacts_w[-1])[1].reshape(-1, 4, 3)
                
                # Set full lenght contact plan
                if len(self.full_length_contact_plan) == 0:
                    # Set next 2 jumps
                    contact_plan = np.concatenate((jump1, jump2), axis=0)
                    repeat_contact_plan = np.repeat(contact_plan, self.gait_horizon, axis=0)
                else:
                    # Add only jump2 to the contact plan
                    repeat_contact_plan = np.repeat(jump2, self.gait_horizon, axis=0)
                        
                self.full_length_contact_plan = np.concatenate(
                    (self.full_length_contact_plan, repeat_contact_plan),
                    axis=0)
                    
            mpc_contacts_w = self.full_length_contact_plan[self.replanning : self.replanning + self.gait_horizon]

            # Update the desired velocity
            i_next_jump = -1
            center_position_next_cnt = np.mean(self.full_length_contact_plan[i_next_jump], axis=0)
            self.v_des = np.round((center_position_next_cnt - q[:3]) / self.gait_period, 2)
            # Scale velocity
            self.v_des *= np.array([1.3, 2., 0.])
            
            self.replanning += 1
            
        return mpc_contacts_w