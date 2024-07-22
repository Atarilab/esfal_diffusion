import numpy as np
import copy
import pinocchio as pin

def apply_symmetry_3d_points(a : np.array, x : int, y : int):
    '''
    apply symmetry to an array of 3D points.
    a [N, 3]
    '''
    a_copy = np.copy(a)
    if len(a.shape) == 2:
        if x < 0 and y < 0:
            a_copy[:, :2] = -a[:, :2]
            
        elif x < 0:
            a_copy[:, 0] = -a[:, 0]

        elif y < 0:
            a_copy[:, 1] = -a[:, 1]
            
    elif len(a.shape) == 1:
        if x < 0 and y < 0:
            a_copy[:2] = -a[:2]
            
        elif x < 0:
            a_copy[0] = -a[0]

        elif y < 0:
            a_copy[1] = -a[1]  
                  
    return a_copy

# FR  -0.0008484878775420449, 0.6875545803138652, -1.331231761101011,
# FL  -0.0044557643132792915, 0.6926241261462901, -1.3313565227202124,
# RR  0.005418005992576224, -0.6904568346955275, 1.330043456714709,
# RL  0.0019115460586124494, -0.6892011722464748, 1.3372704771426844,

ID_SWITCH_X = [3,4,5,0,1,2,9,10,11,6,7,8]
ID_SWITCH_Y = [6,7,8,9,10,11,0,1,2,3,4,5]
ID_SWITCH_XY = [9,10,11,6,7,8,3,4,5,0,1,2]

def apply_symmetry_joints(q : np.array, x : int = 0, y : int = 0):
    '''
    apply symmetry to a joint configuration.
    q : [12]
    [FR, FL, RR, RL]
    '''
    # Switch left and right legs and back front
    q_copy = np.copy(q)
    
    if x < 0 and y < 0:
        q_copy = -q[ID_SWITCH_XY]
        
    # Switch left and right legs
    elif x < 0:
        q_copy = q[ID_SWITCH_X]

    # Switch front and back legs
    elif y < 0:
        q_copy = -q[ID_SWITCH_Y]
        
    return q_copy

def get_symmetric_state(q : np.array, x : int = 0, y : int = 0):
    '''
    apply symmetry to full state configuration.
    q : [3 + 7 + 12 + 6 + 12]
    '''
    q_copy = np.copy(q)

    if x > 0 and y > 0:
        return q_copy
    
    # position is the same
    # symmetry orientation as rpy
    R = pin.Quaternion(np.array(q_copy[3:7])).toRotationMatrix()
    rpy_vector = pin.rpy.matrixToRpy(R)
    rpy_vector = apply_symmetry_3d_points(rpy_vector, x, y)
    R = pin.rpy.rpyToMatrix(rpy_vector)
    q = pin.Quaternion(R)
    q_copy[3:7] = np.array([q.x, q.y, q.z, q.w])
    # joints position
    q_copy[7:7+12] = apply_symmetry_joints(q_copy[7:7+12], x, y)
    # velocities in world frame
    q_copy[19:22] = apply_symmetry_3d_points(q_copy[19:22], x, y)
    q_copy[22:25] = apply_symmetry_3d_points(q_copy[22:25], x, y)
    # joint velocities
    q_copy[-12:] = apply_symmetry_joints(q_copy[-12:], x, y)

    return q_copy



if __name__ == "__main__":
    q0 = np.array([0.11, 0.71, -1.31, 0.12, 0.72, -1.32, -0.13, -0.73, 1.33, -0.14, -0.74, 1.34])
    
    q0_x = apply_symmetry_joints(q0, -1, 0)
    q0_y = apply_symmetry_joints(q0, 0, -1)
    q0_xy = apply_symmetry_joints(q0, -1, -1)
    
    print("q0\n", q0)
    print("q0_x\n", q0_x)
    print("q0_y\n", q0_y)
    print("q0_xy\n", q0_xy)
    
    p = np.random.rand(1, 3)
    print(p)
    p_xy = apply_symmetry_3d_points(p, -1, -1)
    print(p_xy)
    
    state = [1,2,3] + [0,0,0,1] + q0.tolist() + [8,9,10] + q0.tolist()
    state_xy = get_symmetric_state(state, -1, -1)
    print(state_xy)