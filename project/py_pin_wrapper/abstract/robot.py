## this file contains a simulation env with pybullet
## Author : Avadesh Meduri
## Date : 7/05/2021

import numpy as np
import pybullet
from numpy.random import default_rng
from pinocchio.utils import zero

    #from bullet_utils.wrapper import PinBulletWrapper
from robot_properties_solo.config import SoloAbstract, Solo12Config
from py_pin_wrapper.pybullet_env import BulletEnv


class PinBulletWrapper(object):
    """[summary]

    Attributes:
        nq (int): Dimension of the generalized coordiantes.
        nv (int): Dimension of the generalized velocities.
        nj (int): Number of joints.
        nf (int): Number of end-effectors.
        robot_id (int): PyBullet id of the robot.
        pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
        useFixedBase (bool): Determines if the robot base if fixed.
        nb_dof (int): The degrees of freedom excluding the base.
        joint_names (:obj:`list` of :obj:`str`): Names of the joints.
        endeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
    """

    def __init__(
        self, bulletEnv, pinocchio_robot, joint_names, endeff_names, useFixedBase=False
    ):
        """Initializes the wrapper.

        Args:
            robot_id (int): PyBullet id of the robot.
            pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
            joint_names (:obj:`list` of :obj:`str`): Names of the joints.
            endeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
            useFixedBase (bool, optional): Determines if the robot base if fixed.. Defaults to False.
        """
        self._p = bulletEnv._p
        self.nq = pinocchio_robot.nq
        self.nv = pinocchio_robot.nv
        self.nj = len(joint_names)
        self.nf = len(endeff_names)
        self.robot_id = bulletEnv.robots[0]
        self.pinocchio_robot = pinocchio_robot
        self.useFixedBase = useFixedBase
        self.nb_dof = self.nv - 6

        self.joint_names = joint_names
        self.endeff_names = endeff_names

        self.base_linvel_prev = None
        self.base_angvel_prev = None
        self.base_linacc = np.zeros(3)
        self.base_angacc = np.zeros(3)

        # IMU pose offset in base frame
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        self.rng = default_rng()

        self.base_imu_accel_bias = np.zeros(3)
        self.base_imu_gyro_bias = np.zeros(3)
        self.base_imu_accel_thermal = np.zeros(3)
        self.base_imu_gyro_thermal = np.zeros(3)
        self.base_imu_accel_thermal_noise = 0.0001962  # m/(sec^2*sqrt(Hz))
        self.base_imu_gyro_thermal_noise = 0.0000873  # rad/(sec*sqrt(Hz))
        self.base_imu_accel_bias_noise = 0.0001  # m/(sec^3*sqrt(Hz))
        self.base_imu_gyro_bias_noise = 0.000309  # rad/(sec^2*sqrt(Hz))

        bullet_joint_map = {}
        for ji in range(self._p.getNumJoints(self.robot_id)):
            bullet_joint_map[
                self._p.getJointInfo(self.robot_id, ji)[1].decode("UTF-8")
            ] = ji

        self.bullet_joint_ids = np.array(
            [bullet_joint_map[name] for name in joint_names]
        )
        self.pinocchio_joint_ids = np.array(
            [pinocchio_robot.model.getJointId(name) for name in joint_names]
        )

        self.pin2bullet_joint_only_array = []

        if not self.useFixedBase:
            for i in range(2, self.nj + 2):
                self.pin2bullet_joint_only_array.append(
                    np.where(self.pinocchio_joint_ids == i)[0][0]
                )
        else:
            for i in range(1, self.nj + 1):
                self.pin2bullet_joint_only_array.append(
                    np.where(self.pinocchio_joint_ids == i)[0][0]
                )

        # Disable the velocity control on the joints as we use torque control.
        self._p.setJointMotorControlArray(
            self.robot_id,
            self.bullet_joint_ids,
            pybullet.VELOCITY_CONTROL,
            forces=np.zeros(self.nj),
        )

        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = [bullet_joint_map[name] for name in endeff_names]
        self.pinocchio_endeff_ids = [
            pinocchio_robot.model.getFrameId(name) for name in endeff_names
        ]
        #
        self.nb_contacts = len(self.pinocchio_endeff_ids)
        self.contact_status = np.zeros(self.nb_contacts)
        self.contact_forces = np.zeros([self.nb_contacts, 6])

    def get_force(self):
        """Returns the force readings as well as the set of active contacts
        Returns:
            (:obj:`list` of :obj:`int`): List of active contact frame ids.
            (:obj:`list` of np.array((6,1))) List of active contact forces.
        """

        active_contacts_frame_ids = []
        contact_forces = []

        # Get the contact model using the self._p.getContactPoints() api.
        cp = self._p.getContactPoints()

        for ci in reversed(cp):
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]

            if ci[3] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
            elif ci[4] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[4])[0][0]
            else:
                continue

            if self.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
                continue

            active_contacts_frame_ids.append(self.pinocchio_endeff_ids[i])
            force = np.zeros(6)

            force[:3] = (
                normal_force * np.array(contact_normal)
                + lateral_friction_force_1 * np.array(lateral_friction_direction_1)
                + lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            )

            contact_forces.append(force)

        return active_contacts_frame_ids[::-1], contact_forces[::-1]

    def end_effector_forces(self):
        """Returns the forces and status for all end effectors

        Returns:
            (:obj:`list` of :obj:`int`): list of contact status for each end effector.
            (:obj:`list` of np.array(6)): List of force wrench at each end effector
        """
        contact_status = np.zeros(len(self.pinocchio_endeff_ids))
        contact_forces = np.zeros([len(self.pinocchio_endeff_ids), 6])
        # Get the contact model using the pybullet.getContactPoints() api.
        cp = self._p.getContactPoints(self.robot_id)

        for ci in reversed(cp):
            p_ct = np.array(ci[6])
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]
            # Find id
            if ci[3] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
            else:
                continue 
            # Contact active
            contact_status[i] = 1
            contact_forces[i, :3] += (
                normal_force * np.array(contact_normal)
                - lateral_friction_force_1 * np.array(lateral_friction_direction_1)
                - lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            )
            # there are instances when status is True but force is zero, to fix this,
            # we need the below if statement
            if np.linalg.norm(contact_forces[i, :3]) < 1.0e-12:
                contact_status[i] = 0
                contact_forces[i, :3].fill(0.0)
        return contact_status, contact_forces

    def get_base_velocity_world(self):
        """Returns the velocity of the base in the world frame.

        Returns:
            np.array((6,1)) with the translation and angular velocity
        """
        vel, orn = self._p.getBaseVelocity(self.robot_id)
        return np.array(vel + orn).reshape(6, 1)

    def get_base_acceleration_world(self):
        """Returns the numerically-computed acceleration of the base in the world frame.

        Returns:
            np.array((6,1)) vector of linear and angular acceleration
        """
        return np.concatenate((self.base_linacc, self.base_angacc))

    def get_base_imu_angvel(self):
        """Returns simulated base IMU gyroscope angular velocity.

        Returns:
            np.array((3,1)) IMU gyroscope angular velocity (base frame)
        """
        base_inertia_pos, base_inertia_quat = self._p.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            self._p.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = self._p.getBaseVelocity(self.robot_id)

        return (
            self.rot_base_to_imu.dot(rot_base_to_world.T.dot(np.array(base_angvel)))
            + self.base_imu_gyro_bias
            + self.base_imu_gyro_thermal
        )

    def get_base_imu_linacc(self):
        """Returns simulated base IMU accelerometer acceleration.

        Returns:
            np.array((3,1)) IMU accelerometer acceleration (base frame, gravity offset)
        """
        base_inertia_pos, base_inertia_quat = self._p.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            self._p.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = self._p.getBaseVelocity(self.robot_id)

        # Transform the base acceleration to the IMU position, in world frame
        imu_linacc = (
            self.base_linacc
            + np.cross(self.base_angacc, rot_base_to_world @ self.r_base_to_imu)
            + np.cross(
                base_angvel,
                np.cross(base_angvel, rot_base_to_world @ self.r_base_to_imu),
            )
        )

        return (
            self.rot_base_to_imu.dot(
                rot_base_to_world.T.dot(imu_linacc + np.array([0.0, 0.0, 9.81]))
            )
            + self.base_imu_accel_bias
            + self.base_imu_accel_thermal
        )

    def get_state(self):
        """Returns a pinocchio-like representation of the q, dq matrices. Note that the base velocities are expressed in the base frame.

        Returns:
            ndarray: Generalized positions.
            ndarray: Generalized velocities.
        """

        q = zero(self.nq)
        dq = zero(self.nv)

        if not self.useFixedBase:
            (
                base_inertia_pos,
                base_inertia_quat,
            ) = self._p.getBasePositionAndOrientation(self.robot_id)
            # Get transform between inertial frame and link frame in base
            base_stat = self._p.getDynamicsInfo(self.robot_id, -1)
            base_inertia_link_pos, base_inertia_link_quat = self._p.invertTransform(
                base_stat[3], base_stat[4]
            )
            pos, orn = self._p.multiplyTransforms(
                base_inertia_pos,
                base_inertia_quat,
                base_inertia_link_pos,
                base_inertia_link_quat,
            )

            q[:3] = pos
            q[3:7] = orn

            vel, orn = self._p.getBaseVelocity(self.robot_id)
            dq[:3] = vel
            dq[3:6] = orn

            # Pinocchio assumes the base velocity to be in the body frame -> rotate.
            rot = np.array(self._p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            dq[0:3] = rot.T.dot(dq[0:3])
            dq[3:6] = rot.T.dot(dq[3:6])

        # Query the joint readings.
        joint_states = self._p.getJointStates(self.robot_id, self.bullet_joint_ids)

        if not self.useFixedBase:
            for i in range(self.nj):
                q[5 + self.pinocchio_joint_ids[i]] = joint_states[i][0]
                dq[4 + self.pinocchio_joint_ids[i]] = joint_states[i][1]
        else:
            for i in range(self.nj):
                q[self.pinocchio_joint_ids[i] - 1] = joint_states[i][0]
                dq[self.pinocchio_joint_ids[i] - 1] = joint_states[i][1]

        return q, dq

    def get_imu_frame_position_velocity(self):
        """Returns the position and velocity of IMU frame. Note that the velocity is expressed in the IMU frame.

        Returns:
            np.array((3,1)): IMU frame position expressed in world.
            np.array((3,1)): IMU frame velocity expressed in IMU frame.
        """
        base_pose, base_quat = self._p.getBasePositionAndOrientation(self.robot_id)
        base_linvel, base_angvel = self._p.getBaseVelocity(self.robot_id)

        rot_base_to_world = np.array(
            self._p.getMatrixFromQuaternion(base_quat)
        ).reshape((3, 3))
        rot_imu_to_world = rot_base_to_world.dot(self.rot_base_to_imu.T)

        imu_position = base_pose + rot_base_to_world.dot(self.r_base_to_imu)
        imu_velocity = rot_imu_to_world.T.dot(
            base_linvel
            + np.cross(base_angvel, rot_base_to_world.dot(self.r_base_to_imu))
        )
        return imu_position, imu_velocity

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.

        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum

        Args:
          q: Pinocchio generalized position vector.
          dq: Pinocchio generalize velocity vector.
        """
        self.pinocchio_robot.computeJointJacobians(q)
        self.pinocchio_robot.framesForwardKinematics(q)
        self.pinocchio_robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.

        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities)."""
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        """Reset the robot to the desired states.

        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """
        vec2list = lambda m: np.array(m.T).reshape(-1).tolist()

        if not self.useFixedBase:
            # Get transform between inertial frame and link frame in base
            base_stat = self._p.getDynamicsInfo(self.robot_id, -1)
            base_pos, base_quat = self._p.multiplyTransforms(
                vec2list(q[:3]), vec2list(q[3:7]), base_stat[3], base_stat[4]
            )
            self._p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_quat)

            # self._p assumes the base velocity to be aligned with the world frame.
            rot = np.array(self._p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            self._p.resetBaseVelocity(
                self.robot_id, vec2list(rot.dot(dq[:3])), vec2list(rot.dot(dq[3:6]))
            )

            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                self._p.resetJointState(
                    self.robot_id,
                    bullet_joint_id,
                    q[5 + self.pinocchio_joint_ids[i]],
                    dq[4 + self.pinocchio_joint_ids[i]],
                )
        else:
            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                self._p.resetJointState(
                    self.robot_id,
                    bullet_joint_id,
                    q[self.pinocchio_joint_ids[i] - 1],
                    dq[self.pinocchio_joint_ids[i] - 1],
                )

    def send_joint_command(self, tau):
        """Apply the desired torques to the joints.

        Args:
            tau (ndarray): Torque to be applied.
        """
        # TODO: Apply the torques on the base towards the simulator as well.
        if not self.useFixedBase:
            assert tau.shape[0] == self.nv - 6
        else:
            assert tau.shape[0] == self.nv

        zeroGains = tau.shape[0] * (0.0,)

        self._p.setJointMotorControlArray(
            self.robot_id,
            self.bullet_joint_ids,
            pybullet.TORQUE_CONTROL,
            forces=tau[self.pin2bullet_joint_only_array],
            positionGains=zeroGains,
            velocityGains=zeroGains,
        )

    def step_simulation(self):
        """Step the simulation forward."""
        self._p.stepSimulation()

class SoloRobotWrapper(PinBulletWrapper):

    POSITION_NOISE = 7e-3
    ORIENTATION_NOISE = 1e-2
    VELOCITIES_NOISE = 2e-2
    ANG_VELOCITIES_NOISE = 2e-2
    JOINT_POSITION_NOISE = 2e-3
    JOINT_VELOCITIES_NOISE = 5e-3
    
    def __init__(self,
                 config : SoloAbstract = Solo12Config(),
                 q0 : np.array = None,
                 v0 : np.array = None,
                 server = pybullet.GUI,
                 ):
        
        self.config = config
        self.collided = False
        self.robot_id = -1
        self.offset_q = np.array([-0.21, 0., 0.] + [0.]*16)
        
        if q0 is None:
            q0 = np.array(self.config.initial_configuration, dtype=np.float64) + self.offset_q
        if v0 is None:
            v0 = np.array(self.config.initial_velocity, dtype=np.float64)
        self.q0 = q0
        self.v0 = v0
        
        self._init_pin_robot()

        self.eeff_names = [eeff.replace("FOOT", "ANKLE") for eeff in self.config.end_effector_names]
        hip_names = ["FL_HFE", "FR_HFE", "HL_HFE", "HR_HFE"]
        self.shoulder_ids = self.config.shoulder_ids
        self.hip_ids = [self.pin_model.getFrameId(hip_name) for hip_name in hip_names]
        self.eeff_ids = self.config.end_eff_ids
        
        ### Create pybullet server
        self.env = BulletEnv(server)
        robot_id = self.env.add_robot(self)
        self.init_wrapper(self.env)
        
    def init_wrapper(self, bullet_env):
        if self.robot_id == -1:
            self.robot_id = bullet_env.robots[0]
            super(SoloRobotWrapper, self).__init__(
                self.env,
                self.pin_robot,
                self.config.joint_names,
                self.eeff_names
            )
            self.reset_state(self.q0, self.v0)
            
            # TODO Fix this
            self.bullet_endeff_ids = [i - 1 for i in self.bullet_endeff_ids]
            
    def reset_state(self, q : np.ndarray = None, dq : np.ndarray = None):
        if q is None:
            q = np.array(self.config.initial_configuration, dtype=np.float64) + self.offset_q
        if dq is None:
            dq = np.array(self.config.initial_velocity, dtype=np.float64)
        super().reset_state(q, dq)
        
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.centroidalMomentum(q, dq)
        self.update_pinocchio(q, dq)
        return q, dq
        
    def reset_state_with_rand(self, q : np.ndarray = None, dq : np.ndarray = None):
        if q is None:
            q = np.array(self.config.initial_configuration, dtype=np.float64) + self.offset_q
        if dq is None:
            dq = np.array(self.config.initial_velocity, dtype=np.float64)
            
        noise_q, noise_dq = self._get_rand_state_noise()

        q += noise_q
        q[3:7] /= np.linalg.norm(q[3:7]) # normalize quaternion
        dq += noise_dq
        
        self.reset_state(q, dq)
        
        return q, dq
        
    def _get_rand_state_noise(self):
        noise_q = np.concatenate((
            np.random.randn(2,) * SoloRobotWrapper.POSITION_NOISE,
            np.abs(np.random.randn(1,) * SoloRobotWrapper.POSITION_NOISE), # positive height
            np.random.randn(4,) * SoloRobotWrapper.ORIENTATION_NOISE,
            np.random.randn(self.nj) * SoloRobotWrapper.JOINT_POSITION_NOISE,
        ))
        noise_dq = np.concatenate((
            np.random.randn(3,) * SoloRobotWrapper.VELOCITIES_NOISE,
            np.random.randn(3,) * SoloRobotWrapper.ANG_VELOCITIES_NOISE,
            np.random.randn(self.nj) * SoloRobotWrapper.JOINT_VELOCITIES_NOISE,
        ))
        
        return noise_q, noise_dq
        
    def _init_pin_robot(self):
        # Create the robot wrapper in pinocchio.
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.pin_data = self.pin_robot.data
        self.pin_model = self.pin_robot.model
    
    def send_joint_command(self, tau : dict):
        """
        computes the torques using the ID controller and plugs the torques
        Input:
            tau : input torque as a dict {joint name : tau}
        """
        torques_v = np.array([tau[name] for name in self.joint_names])
        super().send_joint_command(torques_v)

    def get_current_contacts(self):
        """
        :return: an array of boolean 1/0 of end-effector current status of contact (0 = no contact, 1 = contact)
        """
        contact_configuration = self.get_force()[0]
        return contact_configuration

    def get_current_eeff_contacts(self):
        """
        :return: an array of boolean 1/0 of end-effector current status of contact (0 = no contact, 1 = contact)
        """
        contact_configuration = self.end_effector_forces()[0]
        return contact_configuration

    def get_frame_location(self, q, dq, frame_idx):
        """
        returns the global location of the frame
        """
        self.pin_robot.framesForwardKinematics(q)
        pos = self.pin_data.oMf[frame_idx].translation
        return pos
    
    def get_foot_locations_world(self) -> np.ndarray:
        """
        returns current foot location of the solo in world frame.
        """
        # Update kinematics
        q, _ = self.get_state()
        self.pin_robot.framesForwardKinematics(q)
        
        # Gather foot locations
        foot_locations = np.empty((self.nf, 3))
        for i, eeff_id in enumerate(self.pinocchio_endeff_ids):
            foot_locations[i, :] = self.pin_data.oMf[eeff_id].translation
        
        return foot_locations
    
    def get_hip_locations_world(self) -> np.ndarray:
        """
        returns current foot location of the solo in world frame.
        """
        # Update kinematics
        q, _ = self.get_state()
        self.pin_robot.framesForwardKinematics(q)
        
        # Gather foot locations
        foot_locations = np.empty((self.nf, 3))
        for i, hip_id in enumerate(self.hip_ids):
            foot_locations[i, :] = self.pin_data.oMf[hip_id].translation
        
        return foot_locations
    
    def is_collision(self, exclude_end_effectors: bool = True) -> bool:
        """
        Return True if some robot geometries are in contact with the environment.
        
        Args:
            - exclude_end_effectors (bool): exclude contacts of the end-effectors.
        """
        is_collision, self.collided = False, False

        # Get all contact points
        contact_points = self._p.getContactPoints(self.robot_id)

        if exclude_end_effectors:
            # Check for collisions excluding end-effectors
            for contact in contact_points:
                if (contact[4] not in self.bullet_endeff_ids and
                    contact[3] not in self.bullet_endeff_ids):
                    is_collision, self.collided = True, True
            return is_collision

        # General collision check
        if contact_points:
            is_collision, self.collided = True, True

        return is_collision
        