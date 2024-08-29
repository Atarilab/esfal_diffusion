## Contains go2 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from mpc_controller.motions.weight_abstract import BiconvexMotionParams

N_JOINTS = 12

#### jump #########################################
trot = BiconvexMotionParams("go2", "Jump")

#########
######### Gait parameters
#########

# Gait horizon
trot.gait_horizon = 1.

# Gait period (s)
trot.gait_period = 0.8
# Gait stance percent [0,1] [FR, FL, RR, RL]
trot.stance_percent = [0.75, 0.75, 0.75, 0.75]
# Gait dt
trot.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
trot.phase_offset = [0., 0.5, 0.5, 0.]
# Gait step height
trot.step_ht = 0.06
# Gait mean/nominal height
trot.nom_ht = 0.32


# Gains toque controller
trot.kp = 13.
trot.kd = 0.1

# ADMM constraints violation norm
trot.rho = 4e+4

#########
######### Kinematic solver
#########

### State
trot.state_wt =  np.array(
    # position (x, y, z)
    [1., 1., 2e2] +
    # orientation (r, p, y)
    [5e3, 5e3, 1e3] +
    # joint positions                    
    [10., 15., 30.]  * 4 +
    # linear velocities (x, y, z)                 
    [10., 10., 2e2] +
    # angular velocities (x, y, z) 
    [3e3, 3e3, 3e2] +
    # joint velocities          
    [10., 35., 45.]  * 4
    )

### Control
trot.ctrl_wt = np.array(
    # force (x, y, z)
    [5e2, 5e2, 1e3] +
    # moment at base (x, y, z)                    
    [1e3, 1e3, 1e3] +
    # torques                 
    [10.0] * N_JOINTS
    )

### Tracking swing end effectors (same for all end effectors swinging)
trot.swing_wt = np.array(
    # contact (x, y, z)
    [2e5, 2e5, 6e4,] +
    # swing (x, y, z)   
    [1e5, 1e5, 4e4,]
    )
### Centroidal
trot.cent_wt = np.array(
    # center of mass (x, y, z)
    3*[1e+1] +
    # linear momentum of CoM (x, y, z)      
    3*[3e+2,] +
    # angular momentum around CoM (x, y, z)       
    3*[7e+2,]         
    )

### Regularization, scale state_wt and ctrl_wt
trot.reg_wt = [
    1.0e-2,
    9.e-6
    ]

#########
######### Dynamics solver
#########

### State:
trot.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e-2, 1e-2, 1e+4] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 4e+2] +
    # angular momentum around CoM (x, y, z)                    
    [6e+4, 6e+4, 2e4] 
    )

### Terminal state:
trot.W_X_ter = np.array(
    # centroidal center of mass (x, y, z)
    [1e+4, 1e+4, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+2, 1e+2, 2e+3] +
    # angular momentum around CoM (x, y, z)                    
    [1e+6, 1e+6, 1e+5]
    )

### Force on each end effectors
trot.W_F = np.array(4*[5.e+0, 5.e+0, 3.e+1])

# Maximum force to apply (will be multiplied by the robot weight)
trot.f_max = np.array([.1, .1, .2])

trot.dyn_bound = np.array(3 * [0.45])

### Orientation correction (weights) modifies angular momentum
trot.ori_correction = [0.3, 0.2, 0.1]