import pybullet as p
import numpy as np
import pinocchio as pin
from bullet_utils.env import BulletEnv

# Constants (set these values accordingly)
UPDATE_VISUALS_STEPS = 10
N_NEXT_CONTACTS = 5
FEET_COLORS = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
SPHERE_RADIUS = 0.015

def desired_contact_locations_callback(env: BulletEnv, sim_step: int, q: np.ndarray, v: np.ndarray, controller) -> None:
    """
    Visualize desired contact locations in PyBullet.

    Args:
        sim_step (int): Simulation step.
        q (np.ndarray): Robot state.
        v (np.ndarray): Robot velocity.
        controller: Controller with gait generation capabilities.
    """
    if sim_step % UPDATE_VISUALS_STEPS == 0:
        # Next contacts in base frame (except height in world frame)
        horizon_step = controller.gait_gen.horizon
        contact_step = max(horizon_step // N_NEXT_CONTACTS, 1)
        next_contacts_B = controller.gait_gen.cnt_plan[::contact_step, :, 1:].reshape(-1, 3)
        all_contact_W = np.empty_like(next_contacts_B)
        
        # Base transform in world frame
        W_T_B = pin.XYZQUATToSE3(q[:7])
        
        if not hasattr(env, 'visuals'):
            env.visuals = []
        
        for i, contacts_B in enumerate(next_contacts_B):
            # Express contact in world frame
            contact_W = W_T_B * contacts_B
            contact_W[-1] = contacts_B[-1]
            all_contact_W[i] = contact_W
            
            # Add visuals
            color = FEET_COLORS[i % len(FEET_COLORS)]
            color[-1] = 0.4 if i > 4 else 1.
            size = SPHERE_RADIUS if i < 4 else SPHERE_RADIUS / 2.
            
            if i < len(env.visuals):
                # Move the existing visual object
                p.resetBasePositionAndOrientation(
                    env.visuals[i],
                    contact_W,
                    p.getQuaternionFromEuler([0.0, 0.0, 0.0])
                )
            else:
                # Create a new visual object
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=size,
                    rgbaColor=color
                )
                visual_body_id = p.createMultiBody(
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=contact_W,
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0])
                )
                env.visuals.append(visual_body_id)
        
        # Remove excess visuals if any
        for i in range(len(next_contacts_B), len(env.visuals)):
            p.removeBody(env.visuals[i])
        env.visuals = env.visuals[:len(next_contacts_B)]
        
        

def position_3d_callback(env: BulletEnv, positions_W: np.ndarray) -> None:
    """
    Visualize positions in PyBullet.

    Args:
        controller: Controller with visuals attribute.
        positions_W (np.ndarray): Array of 3D positions in world coordinates.
    """
    if not hasattr(env, 'visuals'):
        env.visuals = []
    
    for i, pos in enumerate(positions_W):
        # Add visuals
        color = FEET_COLORS[i % len(FEET_COLORS)]
        color[-1] = 0.4 if i > 4 else 1.
        size = SPHERE_RADIUS if i < 4 else SPHERE_RADIUS / 2.
        
        if i < len(env.visuals):
            # Move the existing visual object
            p.resetBasePositionAndOrientation(
                env.visuals[i],
                pos,
                p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            )
        else:
            # Create a new visual object
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=size,
                rgbaColor=color
            )
            visual_body_id = p.createMultiBody(
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pos,
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            )
            env.visuals.append(visual_body_id)
    
    # Remove excess visuals if any
    for i in range(len(positions_W), len(env.visuals)):
        p.removeBody(env.visuals[i])
    env.visuals = env.visuals[:len(positions_W)]
