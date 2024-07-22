## Contains solo 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from robot_properties_solo.config import Solo12Config

N_JOINTS = Solo12Config.buildRobotWrapper().model.nv - 6

#### Bound #######################################
bound = BiconvexMotionParams("solo12", "Bound")

#########
######### Gait parameters
#########

# Gait horizon
bound.gait_horizon = 1.5

# Gait period (s)
bound.gait_period = 0.3
# Gait stance percent [0,1] [FR, FL, RR, RL]
bound.stance_percent = [0.5, 0.5, 0.5, 0.5]
# Gait dt
bound.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
bound.phase_offset = [0.0, 0.0, 0.5, 0.5]
# Gait step height
bound.step_ht = 0.07
# Gait mean/nominal height
bound.nom_ht = 0.25

# Gains torque controller
bound.kp = 3.0
bound.kd = 0.05

# ADMM constraints violation norm
bound.rho = 5e+4

#########
######### Kinematic solver
#########

### State
bound.state_wt = np.array(
    # position (x, y, z)
    [0.0, 0.0, 1e3] +
    # orientation (r, p, y)
    [10.0, 10.0, 10.0] +
    # joint positions                    
    [50.0] * N_JOINTS +
    # linear velocities (x, y, z)                 
    [0.0, 0.0, 0.0] +
    # angular velocities (x, y, z) 
    [100.0, 10.0, 100.0] +
    # joint velocities          
    [0.5] * N_JOINTS
    )

### Control
bound.ctrl_wt = np.array(
    # force (x, y, z)
    [0.5, 0.5, 0.5] +
    # moment at base (x, y, z)                    
    [1.0, 1.0, 1.0] +
    # torques                 
    [0.5] * N_JOINTS
    )

### Tracking swing end effectors (same for all end effectors swinging)
bound.swing_wt = np.array(
    # contact (x, y, z)
    [2e5, 2e5, 1e4,] +
    # swing (x, y, z)   
    [1e4, 1e4, 2e4,]
    )

### Centroidal
bound.cent_wt = np.array(
    # center of mass (x, y, z)
    3*[5e+1] +
    # linear momentum of CoM (x, y, z)      
    3*[3e+2,] +
    # angular momentum around CoM (x, y, z)       
    3*[7e+2,]         
    )

### Regularization, scale state_wt and ctrl_wt
bound.reg_wt = [
    7e-3,
    7e-5
    ]

#########
######### Dynamics solver
#########

### State:
bound.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e-5, 1e-5, 5e+4] +
    # linear momentum of CoM (x, y, z)                    
    [1e1, 1e1, 1e+3] +
    # angular momentum around CoM (x, y, z)                    
    [5e+3, 1e+4, 5e+3] 
    )

### Terminal state:
bound.W_X_ter = 10*np.array(
    # centroidal center of mass (x, y, z)
    [1e-5, 1e-5, 5e+4] +
    # linear momentum of CoM (x, y, z)                    
    [1e1, 1e1, 1e+3] +
    # angular momentum around CoM (x, y, z)                    
    [1e+4, 1e+4, 1e+4]
    )

### Force on each end effectors
bound.W_F = np.array(4*[1e1, 1e+1, 1.5e+1])


# Maximum force to apply (will be multiplied by the robot weight)
bound.f_max = np.array([.3, .2, .4])

bound.dyn_bound = np.array(3 * [0.45])

# Orientation correction (weights) modifies angular momentum
bound.ori_correction = [0.2, 0.8, 0.8]