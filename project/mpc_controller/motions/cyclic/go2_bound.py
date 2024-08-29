## Contains go2 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from mpc_controller.motions.weight_abstract import BiconvexMotionParams

N_JOINTS = 12

#### Bound #########################################
bound = BiconvexMotionParams("go2", "Bound")

#########
######### Gait parameters
#########

# Gait horizon
bound.gait_horizon = 1.5

# Gait period (s)
bound.gait_period = 0.3
# Gait stance percent [0,1] [FR, FL, RR, RL]
bound.stance_percent = [0.5] * 4
# Gait dt
bound.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
bound.phase_offset = [0., 0., 0.5, 0.5]
# Gait step height
bound.step_ht = 0.075
# Gait mean/nominal height
bound.nom_ht = 0.35

# Gains toque controller
bound.kp = 15.
bound.kd = 2.5

# ADMM constraints violation norm
bound.rho = 4e+4

#########
######### Kinematic solver
#########

### State
bound.state_wt =  np.array(
    # position (x, y, z)
    [20., 20., 1e3] +
    # orientation (r, p, y)
    [6e2, 5e2, 8e2] +
    # joint positions                    
    [60.0, 30., 30.]  * 4 +
    # linear velocities (x, y, z)                 
    [10., 10., 1e1] +
    # angular velocities (x, y, z) 
    [6e1, 5e2, 1e3] +
    # joint velocities          
    [50.0, 35., 35.]  * 4
    )

### Control
bound.ctrl_wt = np.array(
    # force (x, y, z)
    [15, 15, 1e3] +
    # moment at base (x, y, z)                    
    [1.0e0, 1.0e0, 1.0e0] +
    # torques                 
    [50.0] * N_JOINTS                      
    )

### Tracking swing end effectors (same for all end effectors swinging)
bound.swing_wt = np.array(
    # contact (x, y, z)
    [1e5, 1e5, 1e4,] +
    # swing (x, y, z)   
    [3e4, 3e4, 2e5,]
    )

### Centroidal
bound.cent_wt = np.array(
    # center of mass (x, y, z)
    3*[5.0e1,] +
    # linear momentum of CoM (x, y, z)                            
    3*[1.0e2,] +
    # angular momentum around CoM (x, y, z)                             
    3*[5.0e2,]                               
    )

### Regularization, scale state_wt and ctrl_wt
bound.reg_wt = [
    1.e-2,
    9.e-6
    ]

#########
######### Dynamics solver
#########

### State:
bound.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e-5, 1e-5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+4, 1e+4, 1e4]                       
    )

### Terminal state:
bound.W_X_ter = np.array(
    # centroidal center of mass (x, y, z)
    [1e+4, 1e+4, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+2, 1e+2, 2e+3] +
    # angular momentum around CoM (x, y, z)                    
    [1e+6, 1e+6, 1e+5]
    )

### Force on each end effectors
bound.W_F = np.array(4*[5e+0, 5e+0, 2.e+1])

# Maximum force to apply (will be multiplied by the robot weight)
bound.f_max = np.array([.5, .5, 2.])

bound.dyn_bound = np.array(3 * [0.45])

### Orientation correction (weights) modifies angular momentum
bound.ori_correction = [0.5, 0.9, 0.4]