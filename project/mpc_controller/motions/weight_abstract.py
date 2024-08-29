## This class is the abstract class to store weights params for different gaits for robots
## Author : Avadesh Meduri & Paarth Shah
## Date : 7/7/21

import numpy as np

class BiconvexMotionParams:

    def __init__(self, robot_name, motion_name, n_joints: int = 12):

        self.robot_name = robot_name
        self.motion_name = motion_name
        self.n_joints = n_joints

        #########
        ######### Gait parameters
        #########

        # Gait horizon
        self.gait_horizon = 1.
        # Gait period (s)
        self.gait_period = 0.5
        # Gait stance percent [0,1] [FR, FL, RR, RL]
        self.stance_percent = [0.5] * 4
        # Gait dt
        self.gait_dt = 0.05
        # Gait offset between legs [0,1] [FR, FL, RR, RL]
        self.phase_offset = [0., 0.5, 0.5, 0.]
        # Gait step height
        self.step_ht = 0.05
        # Gait mean/nominal height
        self.nom_ht = 0.35

        # Gains torque controller
        self.kp = 20.
        self.kd = 0.1
   
        # ADMM constraints violation norm
        self.rho = 5e+4

        #########
        ######### Kinematic solver
        #########

        ### State
        self.state_wt = np.array(
            # position (x, y, z)
            [1., 1., 1.] +
            # orientation (r, p, y)  
            [1., 1., 1.] +
            # joint positions
            [1.] * self.n_joints +
            # linear velocities (x, y, z)
            [1., 1., 1.] +
            # anuglar velocities (x, y, z)
            [1., 1., 1.] +
            # joint velocities
            [1.]  * self.n_joints
            )

        ### Control
        self.ctrl_wt = (
            # force (x, y, z)
            [1., 1., 1.] +
            # moment at base (x, y, z)
            [1., 1., 1.] +
            # torques
            [1.] * self.n_joints  
            )

        ### Tracking swing end effectors (same for all end effectors swinging)
        self.swing_wt = [
            # position (x, y, z)
            [1., 1., 1.] +
            # velocities (x, y, z)     
            [1., 1., 1.]
            ]

        ### Centroidal
        self.cent_wt = [
            # center of mass (x, y, z)
            [1., 1., 1.] +
            # linear momentum of CoM (x, y, z)  
            [1., 1., 1.] +
            # angular momentum around CoM (x, y, z)   
            [1., 1., 1.]
            ]

        ### Regularization, scale state_wt and ctrl_wt
        self.reg_wt = [
            1.,
            1.
            ]

        #########
        ######### Dynamics solver
        #########

        ### State:
        self.W_X = np.array(
            # centroidal center of mass (x, y, z)
            [1., 1., 1.] +
            # linear momentum of CoM (x, y, z)
            [1., 1., 1.] +
            # angular momentum around CoM (x, y, z)
            [1., 1., 1.]   
            )

        ### Terminal state:
        self.W_X_ter = np.array(
            # centroidal center of mass (x, y, z)
            [1., 1., 1.] +
            # linear momentum of CoM (x, y, z)
            [1., 1., 1.] +
            # angular momentum around CoM (x, y, z)
            [1., 1., 1.]  
            )
        
        ### Force penalization on each end effectors
        self.W_F = np.array([1., 1., 1.] * (self.n_joints // 3))

        # Maximum force to apply (will be multiplied by the robot weight)
        self.f_max = np.array([0.4, 0.4, 1.5])

        # Dynamic bounds
        self.dyn_bound = np.array([0.45] * 3)

        ### Orientation correction (weights) modifies angular momentum
        self.ori_correction = [1., 1., 1.]