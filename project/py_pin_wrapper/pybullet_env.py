import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

class BulletEnv(object):
    def __init__(self, server=p.GUI, dt=1.0e-3):
        self.dt = dt
        self.objects = []
        self.visuals = []
        self.robots = []
        
        self._p = bc.BulletClient(connection_mode=server)
        self.connected = True
        self.physicsClientId = self._p._client
        self._p.setGravity(0, 0, -9.81)
        self._p.setPhysicsEngineParameter(fixedTimeStep=dt)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        planeId = self._p.loadURDF("plane.urdf")
        self.objects.append(planeId)
        
    def add_object_from_urdf(
        self, urdf_path, pos=[0, 0, 0], orn=[0, 0, 0, 1], useFixedBase=True
    ):
        """Adds an object described by a URDF file.

        Args:
            urdf_path (str): The absolute path of the URDF file
            pos (list, optional): The initial position of the object in the world frame. Defaults to [0, 0, 0].
            orn (list, optional): The initial orientation of the object in the world frame, expressed in quaternions. Defaults to [0, 0, 0, 1].
            useFixedBase (bool, optional): Determines if the robot base is fixed or not. Defaults to True.

        Returns:
            [int]: The p id of the object if added successfully.
        """
        # Load the object.
        object_id = self._p.loadURDF(urdf_path, useFixedBase=useFixedBase)
        self._p.resetBasePositionAndOrientation(object_id, pos, orn)
        self.objects.append(object_id)
        return object_id
    
    def add_robot(self, robot):
        self._p.setAdditionalSearchPath(robot.config.resources.package_path)
        urdf_path = robot.config.urdf_path
        robotId = self._p.loadURDF(
            urdf_path,
            robot.q0[:3],
            robot.q0[3:7],
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=False,
        )
        
        # Query all the joints.
        num_joints = self._p.getNumJoints(robotId)
        for ji in range(num_joints):
            self._p.changeDynamics(
                robotId,
                ji,
                linearDamping=0.04,
                angularDamping=0.04,
                restitution=0.0,
                lateralFriction=0.5,
                spinningFriction=0.01,
                rollingFriction=0.001
            )
            
        # Camera settings
        self._p.resetDebugVisualizerCamera(1.5, 35., -45., [robot.q0[0], robot.q0[1], 0.])
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.robots.append(robotId)
        
        return robotId

    def disconnect(self):
        if self.connected:
            self._p.resetSimulation()
            self._p.disconnect()
            self.connected = False
            

    def start_video_recording(self, file_name):
        """Starts video recording and save as a mp4 file.

        Args:
            file_name (str): The absolute path of the file to be saved.
        """
        print(file_name)
        self.log_id = self._p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file_name)

    def stop_video_recording(self):
        """Stops video recording if any."""
        if hasattr(self, "log_id"):
            self._p.stopStateLogging(self.log_id)
        
    def __del__(self):
        # Ensure that we disconnect the client when the object is deleted
        self.disconnect()