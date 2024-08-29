import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np

from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mujoco._structs import MjData

UPDATE_VISUALS_STEPS = 50 # Update position every <UPDATE_VISUALS_STEPS> sim steps
FEET_COLORS = [
    [1., 0., 0., 1.], # FR
    [0., 1., 0., 1.], # FL
    [0., 0., 1., 1.], # RR
    [1., 1., 1., 1.], # RL
]
N_NEXT_CONTACTS = 10
SPHERE_RADIUS = 0.012
    
def desired_contact_locations_callback(viewer,
                                       sim_step: int,
                                       q: np.ndarray,
                                       v: np.ndarray,
                                       robot_data: MjData,
                                       controller: ControllerAbstract) -> None:
    
    if sim_step % UPDATE_VISUALS_STEPS == 0: 
        
        # Next contacts in base frame (except height in world frame)
        horizon_step = controller.gait_gen.horizon
        contact_step = max(horizon_step // N_NEXT_CONTACTS, 1)
        next_contacts_B = controller.gait_gen.cnt_plan[::contact_step, :, 1:].reshape(-1, 3)
        all_contact_W = np.empty_like(next_contacts_B)
        
        # Base transform in world frame
        W_T_B = pin.XYZQUATToSE3(q[:7])
        
        viewer.user_scn.ngeom = 0
        for i, contacts_B in enumerate(next_contacts_B):
            
            # Express contact in world frame
            contact_W = W_T_B * contacts_B
            contact_W[-1] = contacts_B[-1]
            all_contact_W[i] = contact_W
            
            # Add visuals
            color = FEET_COLORS[i % len(FEET_COLORS)]
            color[-1] = (0.5 - (i // 4 / (horizon_step)) + 0.5)
            size = SPHERE_RADIUS * (0.7 - (i // 4 / (horizon_step)) + 0.3)

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[size, 0, 0],
                pos=contact_W,
                mat=np.eye(3).flatten(),
                rgba=color,
            )
        viewer.user_scn.ngeom = i + 1
        
        
def position_3d_callback(viewer, positions_W: np.ndarray) -> None:
    
    viewer.user_scn.ngeom = 0
    for i, pos in enumerate(positions_W):

        # Add visuals
        color = FEET_COLORS[i % len(FEET_COLORS)]
        color[-1] = 0.4 if i > 4 else 1.
        size = SPHERE_RADIUS if i < 4 else SPHERE_RADIUS / 2.
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color,
        )
    viewer.user_scn.ngeom = i + 1