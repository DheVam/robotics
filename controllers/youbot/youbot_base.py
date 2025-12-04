import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from transformations import quaternion_conjugate, quaternion_multiply

from controller import Robot


class YouBot(Robot):
    def __init__(self):
        """
        This is your main robot class. It enables access
        to devices such as motors, position sensors, and camera.
        It provides some functions for movement control for:
            - mobile base
            - arm with gripper
            - arm with tray

        You can change/adjust this class as you wish.
        """
        super().__init__()
        
        # basic constants
        self.TIME_STEP = int(self.getBasicTimeStep())  # simulation time step [ms].
        self.WHEEL_RADIUS=0.065  # wheel radius [m].
        self.LX=0.228  # lateral distance from robot's COM to wheel [m].
        self.LY=0.158  # longitudinal distance from robot's COM to wheel [m].
        self.MAX_SPEED=1.5
        
        self.TRAY_ARM = 0
        self.GRIPPER_ARM = 1
        self.ARMS = [self.TRAY_ARM, self.GRIPPER_ARM]
        
        # init mobile base motors and encoders
        self.motors_base = {
            "FL": self.getDevice("wheel2"),
            "FR": self.getDevice("wheel1"),
            "BL": self.getDevice("wheel4"),
            "BR": self.getDevice("wheel3"),
        }
        for key, m in self.motors_base.items():
            m.getPositionSensor().enable(self.TIME_STEP)
            m.setPosition(np.inf)
            m.setVelocity(0.0)
        
        # init tray arm motors and encoders
        self.motors_tray_arm = [
            self.getDevice('trayarm1'),
            self.getDevice('trayarm2'),
            self.getDevice('trayarm3'),
            self.getDevice('trayarm4'),
            self.getDevice('trayarm5'),
        ]
        for m in self.motors_tray_arm:
            m.getPositionSensor().enable(self.TIME_STEP)
            m.setVelocity(0.8)
            
        # init gripper arm
        self.motors_gripper_arm = [
            self.getDevice('arm1'),
            self.getDevice('arm2'),
            self.getDevice('arm3'),
            self.getDevice('arm4'),
            self.getDevice('arm5'),
        ]
        for m in self.motors_gripper_arm:
            m.getPositionSensor().enable(self.TIME_STEP)
            m.setVelocity(0.8)
        
        # init gripper
        self.finger = self.getDevice('finger::left')
        self.finger.setVelocity(0.03)
        
        # load the kinematic chain based on the robot's URDF file
        end_link = 'TCP'  # main link used for forward and inverse kinematics
        camera_link = 'front_rgbd_camera_link'
        tray_link = 'plate'
        URDF_FN = '../../resources/youbot.urdf'
        self.tcp_chain = kp.build_serial_chain_from_urdf(open(URDF_FN), end_link)
        self.camera_chain = kp.build_serial_chain_from_urdf(open(URDF_FN), camera_link)
        self.tray_chain = kp.build_serial_chain_from_urdf(open(URDF_FN), tray_link)
        
        # print chain on console
        print('kinematic chain:')
        print(self.camera_chain)
        
        # finally set up sensors
        # LiDAR
        self.lidar = self.getDevice("Hokuyo UTM-30LX")
        self.lidar.enable(self.TIME_STEP)
        
        # GPS
        self.gps = self.getDevice('gps')
        self.gps.enable(self.TIME_STEP)
        
        # Camera
        self.camera = self.getDevice("rgb_camera")
        self.camera.enable(self.TIME_STEP)
        
        # Depth camera
        self.depth_camera = self.getDevice("depth_camera")
        self.depth_camera.enable(self.TIME_STEP)
        
        # IMU
        self.compass = self.getDevice("compass")
        self.compass.enable(self.TIME_STEP)
        self.accelerometer = self.getDevice("accelerometer")
        self.accelerometer.enable(self.TIME_STEP)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.TIME_STEP)
        
        self.rgb_camera = self.getDevice("rgb_camera")
        self.rgb_camera.enable(self.TIME_STEP)
        self.rgb_camera.recognitionEnable(self.TIME_STEP)
        
        self.depth_camera = self.getDevice("depth_camera")
        self.depth_camera.enable(self.TIME_STEP)
        
    def set_mecanuum_control(self, vx, vy, w):
        """
        A non-blocking behaviour that sets the velocities for the mobile base.
        """
        self.motors_base['FL'].setVelocity(1. / self.WHEEL_RADIUS * (vx - vy - (self.LX + self.LY) * w))
        self.motors_base['FR'].setVelocity(1. / self.WHEEL_RADIUS * (vx + vy + (self.LX + self.LY) * w))
        self.motors_base['BL'].setVelocity(1. / self.WHEEL_RADIUS * (vx + vy - (self.LX + self.LY) * w))
        self.motors_base['BR'].setVelocity(1. / self.WHEEL_RADIUS * (vx - vy + (self.LX + self.LY) * w))
        
    @property
    def arm_home_conf(self):
        return np.array([0, 1.57, -1.635, 1.78, 0.0])

    @property
    def tray_home_conf(self):
        return np.array([0, 1., -2.635, 1.78, 0.0])
    @property
    def tray_home_conf2(self):
        return np.array([-2.4, 1.9, 0.4, -2.3, 1.57])
        
    def close_gripper(self, timeout=1.2):
        """
        blocking behaviour that will close the gripper
        """
        self.finger.setPosition(0.0)
        
        for step in range(int(timeout * 1000) // self.TIME_STEP):
            self.step()
            
    def open_gripper(self, timeout=1.2):
        """
        blocking behaviour that will open the gripper
        """
        self.finger.setPosition(0.025)
            
        for step in range(int(timeout * 1000) // self.TIME_STEP):
            self.step()
        

    def joint_pos(self, arm):
        """
        :param arm: int, 0 for TRAY_ARM and 1 for GRIPPER_ARM
        :return: (6,) ndarray, the current joint position of the robot
        """
        if arm == self.TRAY_ARM:
            motors = self.motors_tray_arm
        elif arm == self.GRIPPER_ARM:
            motors = self.motors_gripper_arm
        else:
            raise ValueError(f'Unknown arm value, must be {self.ARMS}.')
        
        joint_pos = np.asarray([m.getPositionSensor().getValue() for m in motors])
        return joint_pos
        
    def move_PTP(self, arm, target_joint_pos, timeout=10, velocity=0.8):
        """
        synchronised PTP motion
        blocking behaviour, moves the robot to the desired joint position.
        :param arm: int, 0 for TRAY_ARM and 1 for GRIPPER_ARM
        :param target_joint_pos: list/ndarray with joint configuration
        :param timeout: float, timeout in seconds after which this function returns latest
        :param velocity: float, target joint velocity in radians/second
        :return: bool, True if robot reaches the target position
                  else will return False after timeout (in seconds)
        """
        if arm == self.TRAY_ARM:
            motors = self.motors_tray_arm
        elif arm == self.GRIPPER_ARM:
            motors = self.motors_gripper_arm
        else:
            raise ValueError(f'Unknown arm value, must be {self.ARMS}.')
        
        if len(target_joint_pos) != len(motors):
            raise ValueError('target joint configuration has unexpected length')
            
        abs_diffs = np.abs(target_joint_pos - self.joint_pos(arm))
        velocity_gains = abs_diffs / np.max(abs_diffs)
            
        for pos, gain, motor in zip(target_joint_pos, velocity_gains, motors):
            motor.setPosition(pos)
            motor.setVelocity(gain * velocity)
            
        # step through simulation until timeout or position reache
        for step in range(int(timeout * 1000) // self.TIME_STEP):
            self.step()

            # check if the robot is close enough to the target position
            if all(abs(target_joint_pos - self.joint_pos(arm)) < 0.001):
                return True
                
        print('Timeout. Robot has not reached the desired target position (yet).')
        return False
        
    def move_LIN(self, arm, target_ee_pose, velocity=0.1):
        """
        LINEAR motion of end-effector
        blocking behaviour, returns once the movement is completed
        :param arm: int, 0 for TRAY_ARM and 1 for GRIPPER_ARM
        :param target_ee_pose: kinpy.Transform, target pose of end-effector
        :param velocity: float, desired velocity of end-effector in m/s
        """
        if arm == self.TRAY_ARM:
            motors = self.motors_tray_arm
        elif arm == self.GRIPPER_ARM:
            motors = self.motors_gripper_arm
        else:
            raise ValueError(f'Unknown arm value, must be {self.ARMS}.')
            
        # 1st step: interpolate EE waypoints between current and target
        start_pos = self.forward_kinematics(arm).pos
        start_rot = self.forward_kinematics(arm).rot
        
        target_pos = target_ee_pose.pos
        target_rot = target_ee_pose.rot
        
        # determine number of interpolation steps (= time steps in simulation)
        distance = np.linalg.norm(target_pos - start_pos)
        duration = distance / velocity
        dt = self.TIME_STEP / 1000.
        num_steps = int(duration / dt)
        
        if num_steps < 2:
            num_steps = 2
        print(f'move_LIN duration: {duration:.2f}s ({num_steps} steps)')
        
        # set up interpolation of rotation (spherical linear interpolation)
        slerp = Slerp([0, 1], R.from_quat([start_rot, target_rot], scalar_first=True))
            
        # go through waypoints
        for i in range(num_steps + 1):
            t = i / num_steps

            # interpolate position and rotation to get intermediate pose
            intermediate_pos = start_pos + t * (target_pos - start_pos)
            intermediate_rot = slerp(t).as_quat(scalar_first=True)
            
            # use IK to get corresponding joint values
            intermediate_tf = kp.Transform(pos=intermediate_pos, rot=intermediate_rot)
            intermediate_joint_pos = self.inverse_kinematics(arm, intermediate_tf)
            
            # control joints accordingly
            for pos, motor in zip(intermediate_joint_pos, motors):
                motor.setPosition(pos)
                motor.setVelocity(motor.getMaxVelocity())
                
            self.step()
        
    def forward_kinematics(self, arm, joint_pos=None, camera_link=False):
        """
        computes the pose of the chain's end link for given joint position.
        :param arm: int, 0 for TRAY_ARM and 1 for GRIPPER_ARM
        :param joint_pos: joint position for which to compute the end-effector pose
                          if None given, will use the robot's current joint position
        :param camera_link: bool, if True, will give pose of camera
        :return: kinpy.Transform object with pos and rot
        """
        if arm == self.TRAY_ARM:
            if camera_link:
                raise ValueError('Tray arm has no camera link. Cannot compute Forward Kinematics.')
            chain = self.tray_chain
        elif arm == self.GRIPPER_ARM:
            if camera_link:
                chain = self.camera_chain
            else:
                chain = self.tcp_chain
        else:
            raise ValueError(f'Unknown arm value, must be {self.ARMS}.')
            
        if joint_pos is None:
            joint_pos = self.joint_pos(arm)
            
        ee_pose = chain.forward_kinematics(joint_pos)
        return ee_pose
        
    def inverse_kinematics(self, arm, target_pose, camera_link=False):
        """
        Computes a joint configuration to reach the given target pose.
        Note that the resulting joint position might not actually reach the target
        if the target is e.g. too far away.
        :param arm: int, 0 for TRAY_ARM and 1 for GRIPPER_ARM
        :param target_pose: kinpy.Transform, pose of the end link of the chain
        :param camera_link: bool, if True, will use pose of camera
        :return: list/ndarray, joint position
        """
        if arm == self.TRAY_ARM:
            if camera_link:
                raise ValueError('Tray arm has no camera link. Cannot compute Forward Kinematics.')
            chain = self.tray_chain
        elif arm == self.GRIPPER_ARM:
            if camera_link:
                chain = self.camera_chain
            else:
                chain = self.tcp_chain
        else:
            raise ValueError(f'Unknown arm value, must be {self.ARMS}.')
        
        ik_result = chain.inverse_kinematics(target_pose, self.joint_pos(arm))
        
        # check ik_result for plausibility, and try to improve
        
        def get_pose_error(conf):
            # gives tuple of pos error [mm] and orn error [deg]
            ik_pose = chain.forward_kinematics(conf)
            pos_error = np.linalg.norm(ik_pose.pos - target_pose.pos)
            dq = quaternion_multiply(quaternion_conjugate(ik_pose.rot), target_pose.rot)
            orn_error = np.rad2deg(2*np.arctan2(np.linalg.norm(dq[1:]), dq[0]))
            if orn_error > 180:
                orn_error = 360 - orn_error
            return 1000*pos_error, orn_error
        
        additional_tries = 20
        best_ik_result = ik_result
        best_total_error = np.sum(get_pose_error(ik_result))
        
        while (best_total_error > 15) and (additional_tries > 0):
            # retry IK with different initial configuration
            init_conf = np.random.uniform(low=-3.1415, high=3.1415, size=5)
            ik_result = chain.inverse_kinematics(target_pose, init_conf)
            pos_err, orn_err = get_pose_error(ik_result)
            # print(f'{additional_tries}: {pos_err:.2f}mm, {orn_err:.2f}deg, best so far: {best_total_error:.2f}')
            total_err = pos_err + orn_err
            if total_err < best_total_error:
                best_total_error = total_err
                best_ik_result = ik_result
            additional_tries -= 1
            
        # finally, ensure the configuration is within joint limits
        # map too large values by subtracting 2*pi
        # map too small values by adding 2*pi
        best_ik_result = np.where(best_ik_result > np.pi, best_ik_result - 2*np.pi, best_ik_result)
        best_ik_result = np.where(best_ik_result < -np.pi, best_ik_result + 2*np.pi, best_ik_result)
            
        print(f'Found IK result has a pose error of {best_total_error:.2f} [mm]+[deg]')
        return best_ik_result
        
    def get_camera_objects(self):
        """
        Uses the recognition module to detect objects in the image.
        :returns: dict, with object names as key and (u, v) pixel coordinates as value
        """
        detected_objs = {}
        for obj in self.__sup.getDevice("camera").getRecognitionObjects():
            u, v = obj.getPositionOnImage()
            detected_objs[obj.getModel()] = (u, v)
    
        return detected_objs

        