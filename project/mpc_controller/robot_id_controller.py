## This file contains an inverse dynamics controller between two frames
## Author : Avadesh Meduri
## Date : 16/03/2021

import numpy as np
import pinocchio as pin
import time

class InverseDynamicsController():

    def __init__(self, robot, eff_arr, pinModel = None, pinData = None, real_robot = False):
        """
        Input:
            robot : robot object returned by pinocchio wrapper
            eff_arr : end effector name arr
            real_robot : bool true if controller running on real robot
        """
        if pinModel == None:
            self.pin_robot = robot
            self.pinModel = self.pin_robot.model
            self.pinData = self.pin_robot.data
            self.nq = self.pin_robot.nq
            self.nv = self.pin_robot.nv

        else:
            self.pinModel = pinModel
            self.pinData = pinData
            self.nq = pinModel.nq
            self.nv = pinModel.nv

        self.robot_mass = pin.computeTotalMass(self.pinModel)
        self.eff_arr = eff_arr

    def set_gains(self, kp, kd):
        """
        This function is used to set the gains
        Input:
            kp : joint proportional gains
            kd : joint derivative gains
        """
        self.kp = kp
        self.kd = kd

    def compute_id_torques(self, q, v, a):
        """
        This function computes the torques for the give state using rnea
        Input:
            q : joint positions
            v : joint velocity
            a : joint acceleration
        """
        return np.reshape(pin.rnea(self.pinModel, self.pinData, q, v, a), (self.nv,))

    def id_joint_torques(self, q, dq, des_q, des_v, des_a, fff):
        """
        Compute the input torques with gains.
        Input:
            q : joint positions
            dq : joint velocity
            des_q : desired joint positions
            des_v : desired joint velocities
            des_a : desired joint accelerations
            fff : desired feed forward force
        """
        assert len(q) == self.nq

        # Compute inverse dynamics torques
        tau_id = self.compute_id_torques(des_q, des_v, des_a)

        # Initialize effective torque array
        tau_eff = np.zeros(self.nv)

        # Precompute zero velocity part for feedforward force application
        zero_velocity = np.zeros(3)

        for eff_id, f in zip(self.eff_arr, fff.reshape(-1, 3)):
            # Compute Jacobian transpose for the current end-effector
            J = pin.computeFrameJacobian(self.pinModel, self.pinData, des_q,
                                        eff_id, pin.LOCAL_WORLD_ALIGNED).T
            # Compute and accumulate the effective torques
            tau_eff += np.matmul(J, np.hstack((f, zero_velocity)))

        # Calculate joint space control torques
        tau = (tau_id - tau_eff)[6:]
        tau_gain = -self.kp * (q[7:] - des_q[7:]) - self.kd * (dq[6:] - des_v[6:])

        return tau + tau_gain