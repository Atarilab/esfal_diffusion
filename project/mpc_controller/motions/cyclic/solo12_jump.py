## Contains solo12 jump params
## Author : Your Name
## Date : Today's Date

import numpy as np
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from robot_properties_solo.config import Solo12Config

pin_robot = Solo12Config.buildRobotWrapper()
N_JOINTS = pin_robot.nv - 6

#### jump #########################################
jump = BiconvexMotionParams("solo12", "Jump")

#########
######### Gait parameters
#########

# Gait horizon
jump.gait_horizon = 1.0

# Gait period (s)
jump.gait_period = 0.5
# Gait stance percent [0,1] [FR, FL, RR, RL]
jump.stance_percent = [0.4, 0.4, 0.4, 0.4]
# Gait dt
jump.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
jump.phase_offset = [0., 0., 0., 0.]
# Gait step height
jump.step_ht = 0.045
# Gait mean/nominal height
jump.nom_ht = 0.27

# Gains torque controller
jump.kp = 4.5
jump.kd = 0.04

# ADMM constraints violation norm
jump.rho = 5e+4

#########
######### Kinematic solver
#########

### State
jump.state_wt = np.array(
    # position (x, y, z)
    [1.0, 1.0, 500.0] +
    # orientation (r, p, y)
    [2000.0, 2000.0, 2000.0] +
    # joint positions                    
    [1.0, 1.0, 1.0] * (N_JOINTS // 3) +
    # linear velocities (x, y, z)                 
    [1.0, 1.0, 100.0] +
    # angular velocities (x, y, z) 
    [500.0, 500.0, 300.0] +
    # joint velocities          
    [0.5, 0.5, 0.5] * (N_JOINTS // 3)
    )

### Control
jump.ctrl_wt = np.array(
    # force (x, y, z)
    [500.0, 500.0, 800.0] +
    # moment at base (x, y, z)                    
    [1e4, 1e4, 1e4] +
    # torques                 
    [1.0] * N_JOINTS
    )

### Tracking swing end effectors (same for all end effectors swinging)
jump.swing_wt = np.array(
    # contact (x, y, z)
    [1e4, 1e4, 5e3,] +
    # swing (x, y, z)   
    [7e3, 7e3, 1e4,]
    )

### Centroidal
jump.cent_wt = np.array(
    # center of mass (x, y, z)
    3*[1e+0] +
    # linear momentum of CoM (x, y, z)      
    3*[5e+2,] +
    # angular momentum around CoM (x, y, z)       
    3*[5e+2,]         
    )

### Regularization, scale state_wt and ctrl_wt
jump.reg_wt = [
    8.5e-2,
    1e-5
    ]

#########
######### Dynamics solver
#########

### State:
jump.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e+5, 1e+5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+4, 1e+4, 1e4] 
    )

### Terminal state:
jump.W_X_ter = 10*np.array(
    # centroidal center of mass (x, y, z)
    [1e+5, 1e+5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+5, 1e+5, 1e+5]
    )

### Force on each end effectors
jump.W_F = np.array(4*[1.5e+1, 1.5e+1, 2.2e+1])

# Maximum force to apply (will be multiplied by the robot weight)
jump.f_max = np.array([.15, .15, .3])

jump.dyn_bound = np.array(3 * [0.45])

# Orientation correction (weights) modifies angular momentum
jump.ori_correction = [0., 0., 0.1]
