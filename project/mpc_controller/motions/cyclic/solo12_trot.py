import numpy as np
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from robot_properties_solo.config import Solo12Config

N_JOINTS = Solo12Config.buildRobotWrapper().model.nv - 6

#### Trot #########################################
trot = BiconvexMotionParams("solo12", "Trot")

#########
######### Gait parameters
#########

# Gait horizon
trot.gait_horizon = 1.

# Gait period (s)
trot.gait_period = 0.9
# Gait stance percent [0,1] [FR, FL, RR, RL]
trot.stance_percent = [0.84, 0.84, 0.84, 0.84]
# Gait dt
trot.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
trot.phase_offset = [0.0, 0.5, 0.5, 0.0]
# Gait step height
trot.step_ht = 0.045
# Gait mean/nominal height
trot.nom_ht = 0.24

# Gains torque controller
trot.kp = 3.5
trot.kd = 0.04

# ADMM constraints violation norm
trot.rho = 5e+4

#########
######### Kinematic solver
#########

### State
trot.state_wt = np.array(
    # position (x, y, z)
    [1.0, 1.0, 50.0] +
    # orientation (r, p, y)
    [5000.0, 5000.0, 500.0] +
    # joint positions                    
    [1.0, 1.0, 1.0] * (N_JOINTS // 3) +
    # linear velocities (x, y, z)                 
    [50.0, 50.0, 50.0] +
    # angular velocities (x, y, z) 
    [1500.0, 1500.0, 100.0] +
    # joint velocities          
    [0.5, 0.5, 0.5] * (N_JOINTS // 3)
    )


### Control
trot.ctrl_wt = np.array(
    # force (x, y, z)
    [0.0, 0.0, 1000.0] +
    # moment at base (x, y, z)                    
    [5e2, 5e2, 5e2] +
    # torques                 
    [1.0] * N_JOINTS
    )

### Tracking swing end effectors (same for all end effectors swinging)
trot.swing_wt = np.array(
    # contact (x, y, z)
    [1.5e4, 1.5e4, 1e4,] +
    # swing (x, y, z)   
    [1.e4, 1.e4, 1e4,]
    )

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
    5.e-2,
    1e-5
    ]

#########
######### Dynamics solver
#########

### State:
trot.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e-5, 1e-5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+4, 1e+4, 1e4] 
    )

### Terminal state:
trot.W_X_ter = 10*np.array(
    # centroidal center of mass (x, y, z)
    [1e+5, 1e-5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+5, 1e+5, 1e+5]
    )

### Force on each end effectors
trot.W_F = np.array(4*[1e+1, 1e+1, 1.e+1])

# Maximum force to apply (will be multiplied by the robot weight)
trot.f_max = np.array([.5, .5, .5])

trot.dyn_bound = np.array(3 * [0.45])

# Orientation correction (weights) modifies angular momentum
trot.ori_correction = [0., 0., 0.1]