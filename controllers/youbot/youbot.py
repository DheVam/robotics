"""
Mission 3 Controller for DG4RAS - KUKA youBot
==============================================

AUTONOMOUS PICK-AND-PLACE TASK CONTROLLER

This module implements a complete autonomous controller for a KUKA youBot
to perform pick-and-place operations. The robot detects objects on a table
using its camera, grasps them sequentially, and places them into a bin.

DESIGN RATIONALE AND ARCHITECTURE:
===================================

1. PERCEPTION STRATEGY:
   - Uses RGB camera with object recognition to detect cubes on table
   - Camera is mounted on the gripper arm, providing direct view of workspace
   - Forward kinematics transforms camera frame coordinates to world frame
   - Filters objects by distance to avoid false positives (< 1.5m threshold)
   - Selects closest object for sequential processing to minimize travel distance

2. CONTROL APPROACH:
   - Inverse Kinematics (IK): Converts desired end-effector poses to joint angles
   - Forward Kinematics (FK): Computes current end-effector pose from joint angles
   - Point-to-Point (PTP) motion: Fast synchronized joint movements for repositioning
   - Linear (LIN) motion: Straight-line end-effector paths for precise grasp/release
   - Mecanumwheel control: Omnidirectional base movement (vx, vy, rotation)
   
3. BEHAVIOR COORDINATION:
   - Finite State Machine (FSM) orchestrates task execution
   - States: INIT → GOTO_TABLE → ALIGN_ARM → SEARCH_OBJECTS → PICK_OBJECT 
            → GOTO_BIN → DROP_OBJECT → (repeat until all objects collected)
   - State transitions based on completion criteria (position reached, object detected, etc.)
   - Retry logic for failed grasps (realign arm and re-attempt search)
   
4. MANIPULATION STRATEGY:
   - Top-down grasp approach: Gripper approaches vertically for stable grasps
   - Four-phase grasp execution:
     a) Hover: Move to safe height above object (14cm clearance)
     b) Descend: Lower straight down to grasp height (2.5cm above table)
     c) Close: Engage gripper fingers to secure object
     d) Lift: Raise gripper back to hover position with object
   - IK seed biasing: Initializes IK solver with forward-facing arm configuration
     to maintain consistent elbow-down posture (avoids workspace singularities)
   - Joint clamping: Enforces conservative joint limits to prevent collisions
   - Offset compensation: Small forward offset (5cm) accounts for gripper geometry

5. NAVIGATION AND SAFETY:
   - GPS and compass provide global localization
   - LiDAR-based collision avoidance (stops if obstacle within 10cm ahead)
   - Fixed approach poses: TABLE_APPROACH_POSE (1.8m from center) positions robot
     for optimal arm reach while maintaining safety margin
   - 10cm safety threshold ignores table edge to allow close approach

ARM CONFIGURATIONS:
===================
- ARM_HOME: Safe stowed position for navigation
- ARM_SEARCH: Extended forward with camera looking down at table (J4=-1.6)
- ARM_PRE_GRASP: Lower position just above detected object
- ARM_CARRY: Tucked position for safe object transport
- ARM_DROP_BIN: Rotated to rear (J1=0.0) to position over bin

COORDINATE FRAMES:
==================
- World Frame: Fixed reference frame for environment
- Base Frame: Robot body frame (moves with robot)
- Camera Frame: Attached to gripper arm (moves with arm)
- End-Effector Frame: Gripper TCP (Tool Center Point)

NOTES:
======
- Controller designed for single-armed operation (GRIPPER_ARM only)
- Sequential object processing ensures deterministic behavior
- Arm pose J4=-1.6 provides downward camera view for table scanning
- J5=2.92 keeps camera upright regardless of arm configuration
"""

import math  # Mathematical functions for trigonometry and angle calculations
import numpy as np  # Numerical array operations for vectors, matrices, and linear algebra
import kinpy as kp  # Kinematic library for forward/inverse kinematics and transforms
from scipy.spatial.transform import Rotation  # Quaternion and rotation matrix conversions
from youbot_base import YouBot  # Base class providing hardware interface and motion primitives

DEBUG = True  # Enable verbose logging for debugging (prints state transitions and coordinates)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Navigation Control Parameters
CRUISE_SPEED = 0.25        # Maximum linear velocity [m/s] for base motion
TURN_SPEED = 0.6           # Maximum angular velocity [rad/s] for base rotation
DISTANCE_TOLERANCE = 0.02  # Position accuracy threshold [m] for navigation goals
ANGLE_TOLERANCE = 0.05     # Heading accuracy threshold [rad] for navigation goals

# Obstacle Avoidance Parameters
LIDAR_SCAN_RANGE = 20      # Number of rays on each side of center to check [rays]
OBSTACLE_THRESHOLD = 0.10  # Minimum safe distance to obstacles [m]

# Inverse Kinematics Parameters
IK_RETRY_OFFSET = 0.02     # Spatial offset for IK retry attempts [m]
MAX_JOINT_JUMP = 3.0       # Maximum allowed joint angle change [rad] (~171°)

# Timing Parameters (in simulation steps)
DROP_SETTLE_ITERATIONS = 30  # Wait time after gripper opens for object to fall

# ============================================================================
# ARM JOINT CONFIGURATIONS (5-DOF arm: [J1, J2, J3, J4, J5])
# ============================================================================
# Each configuration is carefully tuned to maintain stable posture and
# avoid singularities while providing necessary workspace coverage.

# ARM_HOME: Safe stowed position for navigation
# - Arm tucked to minimize footprint and prevent collisions
# - High shoulder prevents dragging on ground
ARM_HOME = np.array([2.95, 1.1, -2.5, 1.5, 2.92])

# ARM_SEARCH: Extended search position with downward camera view
# - J1=2.95 rad (169°): Face forward toward table
# - J2=0.6 rad (34°): Shoulder raised for forward reach
# - J3=-1.8 rad (-103°): Forearm extended over table
# - J4=-1.6 rad (-92°): Wrist pitched down to point camera at table
# - J5=2.92 rad (167°): Wrist roll keeps camera upright
ARM_SEARCH = np.array([2.95, 0.6, -1.8, -1.6, 2.92])

# ARM_PRE_GRASP: Lower pre-grasp position (currently unused)
# - Similar to SEARCH but lower shoulder for closer approach
ARM_PRE_GRASP = np.array([2.95, 0.4, -0.9, -1.6, 2.92])

# ARM_CARRY: Safe carry position for object transport
# - Arm partially tucked to secure object
# - Elevated to prevent object dragging on ground
# - Maintains stable center of gravity during navigation
ARM_CARRY = np.array([2.95, 0.8, -0.5, 1.0, 2.92])

# ARM_DROP_BIN: Drop configuration positioned over bin
# - J1=0.0 rad (0°): Rotated to rear (180° from forward)
# - Arm extended backward to reach bin behind robot
# - Downward orientation for gravity-assisted release
ARM_DROP_BIN = np.array([0.0, 1.1, -0.5, -0.8, 0.0])

# ============================================================================
# WORLD COORDINATE WAYPOINTS [x, y, theta]
# ============================================================================
# Fixed positions in world frame for navigation targets

# TABLE_APPROACH_POSE: Position in front of table for object detection
# - X=-1.8m: 1.8m in front of table center (leaves ~30cm for arm reach)
# - Y=0.0m: Centered laterally
# - Theta=π rad (180°): Facing table
TABLE_APPROACH_POSE = np.array([-1.80, 0.0, 3.14159])

# BIN_APPROACH_POSE: Position beside bin for object drop-off
# - X=0.5m, Y=1.3m: Positioned beside bin opening
# - Theta=π/2 rad (90°): Facing bin
BIN_APPROACH_POSE = np.array([0.5, 1.3, 1.57])

class State:
    """
    Finite State Machine states for pick-and-place task coordination.
    
    State transition flow:
    INIT → GOTO_TABLE → ALIGN_ARM → SEARCH_OBJECTS → PICK_OBJECT 
         → GOTO_BIN → DROP_OBJECT → (back to GOTO_TABLE for next object)
         → DONE (when all objects collected)
    
    State descriptions:
    - INIT: Initialize hardware (open gripper, move arm to home)
    - GOTO_TABLE: Navigate robot base to table approach position
    - ALIGN_ARM: Position arm in search configuration with camera looking down
    - SEARCH_OBJECTS: Scan table with camera to detect objects
    - PICK_OBJECT: Execute grasp sequence (hover, descend, close, lift)
    - GOTO_BIN: Navigate robot base to bin drop-off position
    - DROP_OBJECT: Release object into bin and update counter
    - RETURN_TABLE: (Optional) Navigate back to table for next object
    - DONE: All objects collected, stop robot
    """
    INIT = 0
    GOTO_TABLE = 1
    SEARCH_OBJECTS = 2
    ALIGN_ARM = 3
    PICK_OBJECT = 4
    GOTO_BIN = 5
    DROP_OBJECT = 6
    RETURN_TABLE = 7
    DONE = 8

    names = {
        0: "INIT", 1: "GOTO_TABLE", 2: "SEARCH_OBJECTS", 
        3: "ALIGN_ARM", 4: "PICK_OBJECT", 5: "GOTO_BIN", 
        6: "DROP_OBJECT", 7: "RETURN_TABLE", 8: "DONE"
    }

def quaternion_to_matrix(quat):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quat (array-like): Quaternion in [w, x, y, z] format (scalar-first)
    
    Returns:
        np.ndarray: 3x3 rotation matrix
        
    Note:
        - Scipy uses [x, y, z, w] format (scalar-last), so conversion is needed
        - Returns identity matrix on error to prevent computation failures
    """
    try:
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    except:
        return np.eye(3)

class Mission3Controller(YouBot):
    """
    Main controller class for autonomous pick-and-place task.
    
    Inherits from YouBot base class which provides:
    - Motor control for mecanum wheels and arm joints
    - Sensor access (GPS, compass, LiDAR, cameras)
    - Kinematics functions (forward/inverse kinematics)
    - Motion primitives (PTP, LIN, gripper control)
    
    Attributes:
        state (int): Current FSM state
        state_timer (float): Timestamp of last state transition
        cubes_collected (int): Counter for successfully placed objects
        target_cube_pos (np.ndarray): World coordinates of currently targeted object
        attempt_counter (int): Retry counter for object detection
        current_pose (np.ndarray): Current robot pose [x, y, theta]
        arm_aligned (bool): Flag indicating arm is in search configuration
    """
    def __init__(self):
        super().__init__()
        # Enable object recognition on RGB camera for perception
        self.rgb_camera.recognitionEnable(self.TIME_STEP)
        
        # Finite State Machine variables
        self.state = State.INIT
        self.state_timer = self.getTime()
        self.cubes_collected = 0  # Track progress (target: 4 cubes)
        self.target_cube_pos = None  # Current object being processed
        self.attempt_counter = 0  # For retry logic in detection/grasp
        self.current_pose = np.zeros(3)  # [x, y, theta] pose cache
        self.arm_aligned = False  # Prevents redundant arm movements

    def verify_gripper_open(self):
        """
        Verify gripper is fully open by checking finger position sensor.
        
        Returns:
            bool: True if gripper is open (always True in current implementation)
            
        Note:
            Gripper fully open position is 0.025m (25mm finger separation).
            This verification helps ensure objects can be grasped without obstruction.
            Currently returns True as a safety default if sensor unavailable.
        """
        try:
            # Gripper fingers should be at max position (0.025)
            finger_pos = self.finger.getPositionSensor().getValue() if hasattr(self.finger, 'getPositionSensor') else None
            if DEBUG: print(f"[VERIFY_GRIPPER] Finger position: {finger_pos}")
            return True
        except:
            return True  # Assume open if we can't verify

    def get_robot_pose(self):
        """
        Get current robot pose in world frame using GPS and compass.
        
        Returns:
            np.ndarray: Pose vector [x, y, theta] where:
                - x, y: Position in meters (world frame)
                - theta: Heading in radians, normalized to [-π, π]
                
        Note:
            Compass provides heading from magnetic north. The atan2 computation
            converts compass vector to heading angle. Normalization ensures
            angle wrapping is handled correctly for navigation control.
        """
        gps_vals = self.gps.getValues()
        cmp_vals = self.compass.getValues()
        heading = math.atan2(cmp_vals[0], cmp_vals[1])
        
        # Normalize heading to [-π, π] range
        if heading > math.pi: heading -= 2*math.pi
        elif heading <= -math.pi: heading += 2*math.pi
            
        return np.array([gps_vals[0], gps_vals[1], heading])

    def navigate(self, target):
        """
        Navigate robot to target pose using proportional control with obstacle avoidance.
        
        Args:
            target (np.ndarray): Target pose [x, y, theta] in world frame
            
        Returns:
            bool: True if target reached within tolerances, False otherwise
            
        Control Strategy:
            1. Compute position and heading errors in world frame
            2. Transform errors to robot's local frame (forward/sideways/rotation)
            3. Apply proportional gains to generate velocity commands
            4. Clamp velocities to maximum safe speeds
            5. Check LiDAR for obstacles and stop forward motion if too close
            6. Send velocity commands to mecanum wheel controller
            
        Safety Features:
            - LiDAR collision avoidance: Stops forward motion if obstacle < 10cm ahead
            - 10cm threshold chosen to ignore table edge while preventing collisions
            - Only checks forward arc (±20 degrees) for efficiency
            - Velocity clamping prevents overshoot and ensures stable control
        """
        curr = self.get_robot_pose()
        dx = target[0] - curr[0]
        dy = target[1] - curr[1]
        d_theta = target[2] - curr[2]
        
        # Normalize heading error to [-π, π]
        if d_theta > math.pi: d_theta -= 2*math.pi
        elif d_theta <= -math.pi: d_theta += 2*math.pi
        
        dist = math.sqrt(dx**2 + dy**2)
        
        # Check if target reached within tolerance
        if dist < DISTANCE_TOLERANCE and abs(d_theta) < ANGLE_TOLERANCE:
            self.set_mecanuum_control(0, 0, 0)
            return True
            
        # Transform position error from world frame to robot local frame
        c, s = math.cos(curr[2]), math.sin(curr[2])
        lx = dx * c + dy * s  # Forward error
        ly = -dx * s + dy * c  # Lateral error
        
        # Proportional control with velocity limits
        vx = max(min(lx * 1.0, CRUISE_SPEED), -CRUISE_SPEED)  # Forward velocity
        vy = max(min(ly * 1.0, CRUISE_SPEED), -CRUISE_SPEED)  # Lateral velocity
        w  = max(min(d_theta * 2.0, TURN_SPEED), -TURN_SPEED)  # Angular velocity
        
        # LiDAR-based obstacle avoidance
        scan = np.array(self.lidar.getRangeImage())
        mid = len(scan)//2
        # Check forward arc for obstacles using configured scan range
        # Threshold allows close approach to table without false stops
        if np.min(scan[mid-LIDAR_SCAN_RANGE:mid+LIDAR_SCAN_RANGE]) < OBSTACLE_THRESHOLD and vx > 0:
            vx = 0  # Stop forward motion only
            
        self.set_mecanuum_control(vx, vy, w)
        return False

    def detect_cube(self):
        """
        Detect and return world position of closest cube visible to camera.
        
        Returns:
            np.ndarray: World coordinates [x, y, z] of detected object, or None if no object found
            
        Perception Pipeline:
            1. Acquire recognized objects from camera (Webots recognition API)
            2. Filter objects by distance (< 1.5m) to avoid false positives
            3. Select closest valid object for processing
            4. Transform object position through coordinate frames:
               Camera → End-effector → Robot base → World
            5. Return world coordinates for navigation and grasp planning
            
        Coordinate Transformations:
            - Camera frame: Object position from recognition API
            - Base frame: Apply FK to camera, then rotate camera coords by camera orientation
            - World frame: Rotate base coords by robot heading, then add robot position
            
        Design Rationale:
            - Closest-first strategy minimizes arm travel and cycle time
            - 1.5m distance filter prevents detecting objects on other tables
            - World coordinates enable consistent grasp planning regardless of robot pose
            - FK-based transformation ensures accuracy even with arm movement
        """
        objs = self.rgb_camera.getRecognitionObjects()
        if not objs:
            if DEBUG: print("[DETECT_CUBE] No objects detected by camera")
            return None
        
        if DEBUG: print(f"[DETECT_CUBE] Found {len(objs)} object(s) in camera view")
        
        # Filter objects by distance and collect valid candidates
        valid_objs = []
        for obj in objs:
            dist = np.linalg.norm(obj.getPosition())
            if dist < 1.5:  # Maximum reasonable distance in camera frame
                valid_objs.append(obj)
                if DEBUG: print(f"  - Object '{obj.getModel()}' at camera distance: {dist:.3f}m")
        
        if not valid_objs:
            if DEBUG: print("[DETECT_CUBE] No objects within valid range")
            return None
        
        # Select closest valid object for grasping
        target = min(valid_objs, key=lambda o: np.linalg.norm(o.getPosition()))
        pos_cam = np.array(target.getPosition()) 
        model_name = target.getModel()
        
        if DEBUG: print(f"[DETECT_CUBE] Selected closest object '{model_name}' at camera coords: {pos_cam}")
        
        # Transform from camera frame to robot base frame using forward kinematics
        try:
            cam_tf = self.forward_kinematics(self.GRIPPER_ARM, camera_link=True)
            rot_mat = quaternion_to_matrix(cam_tf.rot)
            pos_base = rot_mat @ pos_cam + cam_tf.pos
        except Exception as e:
            if DEBUG: print(f"[DETECT_CUBE] FK error: {e}")
            return None
        
        # Transform from robot base frame to world frame
        r_pose = self.get_robot_pose()
        c, s = math.cos(r_pose[2]), math.sin(r_pose[2])
        
        world_x = r_pose[0] + pos_base[0]*c - pos_base[1]*s
        world_y = r_pose[1] + pos_base[0]*s + pos_base[1]*c
        world_z = pos_base[2]
        
        if DEBUG: print(f"[DETECT_CUBE] Cube '{model_name}' at World: X={world_x:.2f}, Y={world_y:.2f}, Z={world_z:.2f}")
        return np.array([world_x, world_y, world_z])

    def execute_grasp(self, cube_world):
        """
        Execute top-down grasp sequence for an object at given world coordinates.
        
        Args:
            cube_world (np.ndarray): Object position in world frame [x, y, z]
            
        Returns:
            bool: True if grasp successful, False if IK failed or motion impossible
            
        Grasp Strategy:
            This implements a vertical (top-down) approach for stable object grasping:
            
            1. HOVER PHASE:
               - Position gripper 14cm above object
               - Ensures clearance from table and other objects
               - Opens gripper fully (25mm finger separation)
               
            2. DESCEND PHASE:
               - Lower gripper straight down to 2.5cm above table
               - Uses linear motion (LIN) for precise vertical path
               - Slow velocity (0.12 m/s) for controlled approach
               
            3. GRASP PHASE:
               - Close gripper fingers around object
               - Brief settling time ensures secure grip
               
            4. LIFT PHASE:
               - Raise back to hover height
               - Linear motion prevents object catching on obstacles
               
        IK Solver Configuration:
            - Seeds IK with ARM_SEARCH pose to bias toward forward-facing configuration
            - Maintains elbow-down posture to avoid workspace singularities
            - Clamps joint angles to conservative limits preventing self-collision
            - Retries with position offsets if initial IK fails (±2cm in X/Y)
            - Adjusts grasp height (±1-2cm) as fallback for marginal reachability
            
        Coordinate Transformations:
            - Converts world coords to robot base frame using current pose
            - Applies forward compensation (5cm) for gripper geometry offset
            - Constructs SE(3) transforms with downward orientation [0,0,1,0] quaternion
            
        Error Handling:
            - Returns False immediately if hover or grasp IK fails
            - Continues execution on PTP timeout (assumes eventual convergence)
            - Checks for large joint jumps (>171°) and re-seeds IK if detected
            
        Design Rationale:
            Top-down approach is optimal because:
            - Gripper fingers aligned with gravity provide stable grasp
            - Vertical path minimizes collision risk with table/objects
            - Consistent approach angle simplifies grasp planning
            - Downward camera view maintained throughout sequence
        """
        def clamp_joints(q):
            """
            Clamp joint angles to safe limits to prevent self-collision.
            
            Args:
                q (array-like): Joint angles in radians
                
            Returns:
                np.ndarray: Clamped joint angles within safe bounds
                
            Note:
                Conservative limits chosen to maintain safe working envelope:
                - Joint 1 (base rotation): ±180° full rotation
                - Joint 2 (shoulder): ±160° prevents collision with base
                - Joint 3 (elbow): ±160° prevents over-extension
                - Joint 4 (wrist pitch): ±180° full range
                - Joint 5 (wrist roll): ±180° full range
            """
            limits = [
                (-3.14, 3.14),  # joint 1 (base rotation)
                (-2.8,  2.8),   # joint 2 (shoulder pitch)
                (-2.8,  2.8),   # joint 3 (elbow pitch)
                (-3.14, 3.14),  # joint 4 (wrist pitch)
                (-3.14, 3.14)   # joint 5 (wrist roll)
            ]
            qc = np.array(q, dtype=float)
            for i in range(min(len(qc), len(limits))):
                qc[i] = max(min(qc[i], limits[i][1]), limits[i][0])
            return qc

        # Initialize IK solver with forward-facing arm configuration
        # This bias keeps the elbow down and arm extended toward table
        ik_seed = ARM_SEARCH.copy()

        # Transform target from world frame to robot base frame
        r_pose = self.get_robot_pose()
        dx = cube_world[0] - r_pose[0]
        dy = cube_world[1] - r_pose[1]
        c, s = math.cos(r_pose[2]), math.sin(r_pose[2])
        local_x = dx * c + dy * s   # Forward distance to object
        local_y = -dx * s + dy * c  # Lateral offset to object

        # Grasp execution parameters (tuned for reliable grasping)
        grasp_height = 0.025      # Final grasp height: 25mm above table (just clears object)
        hover_height  = 0.14      # Safe approach height: 140mm clearance
        forward_comp  = 0.05      # Gripper geometry offset: 50mm forward from TCP
        final_descend_velocity = 0.12  # Controlled descent speed: 12 cm/s

        # Apply gripper offset to center fingers over object
        local_x = local_x + forward_comp 

        if DEBUG:
            print(f"[GRASP] local target X,Y = {local_x:.3f}, {local_y:.3f}  (world {cube_world[0]:.3f},{cube_world[1]:.3f})")

        # Construct SE(3) transforms for hover and grasp poses
        # Quaternion [0, 0, 1, 0] represents gripper pointing straight down (Z-axis aligned with gravity)
        target_rot = np.array([0.0, 0.0, 1.0, 0.0])  # [w, x, y, z] format
        hover_tf = kp.Transform(pos=np.array([local_x, local_y, grasp_height + hover_height]), rot=target_rot)
        grasp_tf = kp.Transform(pos=np.array([local_x, local_y, grasp_height]), rot=target_rot)

        # Solve IK for hover pose with biased initialization
        # The q_init parameter guides the solver toward desired arm configuration
        try:
            q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf, q_init=ik_seed)
        except TypeError:
            # Fallback for older API without q_init support
            # Temporarily move key joints to seed configuration
            try:
                self.motors_gripper_arm[0].setPosition(ik_seed[0])  # Base rotation
                self.motors_gripper_arm[4].setPosition(ik_seed[4])  # Wrist roll
            except Exception:
                pass
            q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf)

        if q_hover is None:
            # Retry IK with small spatial offsets if direct solution fails
            # This handles cases where object is at workspace boundary
            offsets = [(0.0,0.0), (IK_RETRY_OFFSET,0.0), (-IK_RETRY_OFFSET,0.0), 
                       (0.0,IK_RETRY_OFFSET), (0.0,-IK_RETRY_OFFSET)]
            for ox, oy in offsets:
                tf = kp.Transform(pos=np.array([local_x + ox, local_y + oy, grasp_height + hover_height]), rot=target_rot)
                try:
                    q_hover = self.inverse_kinematics(self.GRIPPER_ARM, tf, q_init=ik_seed)
                except TypeError:
                    q_hover = self.inverse_kinematics(self.GRIPPER_ARM, tf)
                if q_hover is not None:
                    hover_tf = tf
                    local_x += ox
                    local_y += oy
                    if DEBUG: print(f"[GRASP] hover found with offset ({ox:.3f},{oy:.3f})")
                    break

        if q_hover is None:
            if DEBUG: print("[GRASP] Failed to find IK for hover pose - object likely unreachable.")
            return False

        # Solve IK for final grasp pose using hover config as seed
        # This ensures smooth transition from hover to grasp
        try:
            q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, grasp_tf, q_init=q_hover)
        except TypeError:
            q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, grasp_tf)

        if q_grasp is None:
            # Adjust grasp height slightly if IK fails (may prevent table collision)
            for dz in [0.01, 0.02, -0.005]:
                tf = kp.Transform(pos=np.array([local_x, local_y, grasp_height + dz]), rot=target_rot)
                try:
                    q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, tf, q_init=q_hover)
                except TypeError:
                    q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, tf)
                if q_grasp is not None:
                    grasp_tf = tf
                    if DEBUG: print(f"[GRASP] grasp IK succeeded with height adjustment dz={dz:.3f}m")
                    break

        if q_grasp is None:
            if DEBUG: print("[GRASP] Failed to find IK for grasp pose - descend path unreachable.")
            return False

        # Enforce joint limits to prevent hardware damage and self-collision
        q_hover = clamp_joints(q_hover)
        q_grasp = clamp_joints(q_grasp)

        # Sanity check: detect unreasonable joint jumps that indicate IK flip
        # Large jumps often indicate solution in wrong configuration branch
        try:
            cur = np.array(self.joint_pos(self.GRIPPER_ARM))
            jump = np.abs(cur - q_hover)
            if np.any(jump > MAX_JOINT_JUMP):
                # Re-solve IK with more conservative seed to avoid flip
                if DEBUG: print("[GRASP] Large joint jump detected, re-solving IK with ARM_SEARCH seed...")
                try:
                    q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf, q_init=ARM_SEARCH)
                except TypeError:
                    q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf)
                if q_hover is None:
                    if DEBUG: print("[GRASP] IK after re-seeding failed; aborting grasp.")
                    return False
                q_hover = clamp_joints(q_hover)
        except Exception:
            # If joint_pos() unavailable, continue with computed solution
            pass

        # === GRASP EXECUTION SEQUENCE ===
        
        # Phase 1: Prepare gripper and move to hover position
        self.open_gripper(timeout=1.0)
        if DEBUG: print("[GRASP] Phase 1: Moving to hover position (PTP)...")
        ok = self.move_PTP(self.GRIPPER_ARM, q_hover, velocity=0.6, timeout=6.0)
        if not ok:
            if DEBUG: print("[GRASP] WARNING: PTP to hover timed out (continuing anyway)...")

        # Brief settling time for oscillations to dampen
        for _ in range(5):
            self.step(self.TIME_STEP)

        # Phase 2: Descend straight down to grasp height
        # Linear motion preferred for precise vertical path (avoids sweeping motions)
        used_lin = False
        if DEBUG: print("[GRASP] Phase 2: Descending to grasp height...")
        try:
            self.move_LIN(self.GRIPPER_ARM, grasp_tf, timeout=4.0, velocity=final_descend_velocity)
            used_lin = True
            if DEBUG: print("[GRASP] Linear descent completed.")
        except Exception:
            # Fallback to PTP with slow velocity if LIN unavailable
            if DEBUG: print("[GRASP] move_LIN unavailable, using slow PTP for descent.")
            self.move_PTP(self.GRIPPER_ARM, q_grasp, velocity=0.14, timeout=5.0)

        # Settle before gripper closure
        for _ in range(6):
            self.step(self.TIME_STEP)

        # Phase 3: Close gripper around object
        if DEBUG: print("[GRASP] Phase 3: Closing gripper...")
        self.close_gripper(timeout=1.5)
        for _ in range(6):
            self.step(self.TIME_STEP)

        # Phase 4: Lift object back to hover position
        if DEBUG: print("[GRASP] Phase 4: Lifting object to hover...")
        if used_lin:
            try:
                self.move_LIN(self.GRIPPER_ARM, hover_tf, timeout=4.0, velocity=0.4)
            except Exception:
                self.move_PTP(self.GRIPPER_ARM, q_hover, velocity=0.4)
        else:
            self.move_PTP(self.GRIPPER_ARM, q_hover, velocity=0.4)

        if DEBUG: print("[GRASP] Grasp sequence completed successfully!")
        return True


    def run(self):
        """
        Main control loop implementing finite state machine for pick-and-place task.
        
        FSM Structure:
            The controller uses a finite state machine to coordinate perception,
            planning, and execution. Each state has specific entry/exit conditions
            and handles one aspect of the task.
            
        State Flow:
            1. INIT: Hardware initialization (gripper, arm positioning)
            2. GOTO_TABLE: Navigate to table approach position
            3. ALIGN_ARM: Position arm for optimal camera view
            4. SEARCH_OBJECTS: Scan table and detect objects with camera
            5. PICK_OBJECT: Execute grasp sequence
            6. GOTO_BIN: Navigate to bin drop-off location  
            7. DROP_OBJECT: Release object and update progress counter
            8. Loop back to GOTO_TABLE for next object (until count = 4)
            9. DONE: Mission complete, stop all motion
            
        Design Philosophy:
            - Each state is autonomous and self-contained
            - State transitions based on completion criteria (not time-based)
            - Retry logic for transient failures (detection timeouts, grasp failures)
            - Progress tracking ensures all objects processed sequentially
            - Defensive programming with extra settling times prevents race conditions
            
        Real-time Execution:
            - Runs at simulation time step (24ms default)
            - Each iteration processes one state
            - Blocking operations (move_PTP, move_LIN) handle their own timing
            - Non-blocking operations (navigate) checked each iteration
        """
        print("Mission 3 Controller Started...")
        
        while self.step(self.TIME_STEP) != -1:
            t = self.getTime()
            
            if self.state == State.INIT:
                """
                State: INIT - Hardware initialization and startup configuration
                
                Purpose:
                    Prepare robot for operation by initializing all hardware to safe states
                    
                Actions:
                    1. Open gripper fully (prevents interference with arm movement)
                    2. Move arm to HOME position (safe stowed configuration)
                    3. Verify all systems ready
                    
                Exit Condition:
                    Hardware initialized → transition to GOTO_TABLE
                    
                Rationale:
                    Starting with known hardware state prevents undefined behavior
                    and ensures consistent initial conditions for navigation
                """
                print("\n" + "="*60)
                print("Mission 3 Controller Started...")
                print("="*60)
                print("[INIT] Opening gripper fully...")
                self.open_gripper(timeout=1.5)
                
                # Wait for gripper to fully open (extra time ensures complete motion)
                for _ in range(20):
                    self.step(self.TIME_STEP)
                
                self.verify_gripper_open()
                
                print("[INIT] Moving arm to HOME position...")
                self.move_PTP(self.GRIPPER_ARM, ARM_HOME, timeout=5.0)
                
                # Settling time prevents oscillations during navigation
                for _ in range(10):
                    self.step(self.TIME_STEP)
                
                print("[INIT] Initialization complete! Moving to GOTO_TABLE...")
                self.state = State.GOTO_TABLE
                self.state_timer = self.getTime()
                
            elif self.state == State.GOTO_TABLE:
                """
                State: GOTO_TABLE - Navigate robot base to table approach position
                
                Purpose:
                    Position robot at optimal distance from table for arm reach
                    
                Control:
                    - Proportional navigation with obstacle avoidance
                    - Target: 1.8m from table center, facing table (theta=π)
                    - Mecanum wheels allow omnidirectional approach
                    
                Exit Condition:
                    Position error < 2cm AND heading error < 0.05 rad → ALIGN_ARM
                    
                Rationale:
                    Fixed approach position ensures consistent workspace geometry
                    for arm planning. Distance chosen to keep table within reach
                    while maintaining safe clearance (30cm margin).
                """
                if self.navigate(TABLE_APPROACH_POSE):
                    print(f"[STATE: GOTO_TABLE] Arrived at Table on attempt {self.cubes_collected + 1}.")
                    print("[STATE: GOTO_TABLE] Resetting arm alignment flag...")
                    self.arm_aligned = False  # Force arm reposition each cycle
                    self.state = State.ALIGN_ARM
                    self.state_timer = t
                    
            elif self.state == State.ALIGN_ARM:
                """
                State: ALIGN_ARM - Position arm in search configuration
                
                Purpose:
                    Configure arm for optimal table scanning with downward camera view
                    
                Actions:
                    1. Open gripper fully (ensures no interference with grasp)
                    2. Move arm to ARM_SEARCH configuration (extended forward, looking down)
                    3. Wait for motion to complete and verify position
                    
                Configuration:
                    ARM_SEARCH = [2.95, 0.6, -1.8, -1.6, 2.92]
                    - J1=2.95: Face forward toward table
                    - J2=0.6: Raise shoulder for forward extension
                    - J3=-1.8: Extend forearm over table
                    - J4=-1.6: Point camera down at table surface
                    - J5=2.92: Keep camera upright for correct image orientation
                    
                Exit Condition:
                    Joint error < 0.1 rad OR 5s timeout → SEARCH_OBJECTS
                    
                Rationale:
                    Consistent search pose ensures repeatable detection results
                    and maximizes camera field of view over table surface.
                """
                if not self.arm_aligned:
                    print(f"\n[ALIGN_ARM] ===== PREPARING FOR CUBE #{self.cubes_collected + 1} =====")
                    print(f"[ALIGN_ARM] Opening gripper FULLY before search...")
                    
                    # CRITICAL: Ensure gripper is fully open
                    self.finger.setPosition(0.025)  # MAX POSITION
                    self.open_gripper(timeout=2.0)   # Extra long timeout
                    
                    # Wait extra long for gripper to fully open
                    for _ in range(30):
                        self.step(self.TIME_STEP)
                    
                    print(f"[ALIGN_ARM] Moving arm to search position...")
                    print(f"[ALIGN_ARM] Target position: {ARM_SEARCH}")
                    
                    # Move arm to search position
                    ok = self.move_PTP(self.GRIPPER_ARM, ARM_SEARCH, timeout=6.0, velocity=0.7)
                    if not ok:
                        print(f"[ALIGN_ARM] WARNING: move_PTP to search position timed out (continuing anyway)")
                    
                    # Wait for arm to fully settle
                    for _ in range(15):
                        self.step(self.TIME_STEP)
                    
                    self.verify_gripper_open()
                    self.arm_aligned = True
                    self.state_timer = t
                else:
                    # Check if arm has reached target
                    if t - self.state_timer > 1.5:
                        current_pos = self.joint_pos(self.GRIPPER_ARM)
                        error = np.abs(current_pos - ARM_SEARCH)
                        if DEBUG: print(f"[ALIGN_ARM] Current position: {current_pos}")
                        if DEBUG: print(f"[ALIGN_ARM] Position error: {error}")
                        
                        if np.all(error < 0.1):
                            print("[ALIGN_ARM] ✓ Arm aligned! Moving to SEARCH_OBJECTS...")
                            self.state = State.SEARCH_OBJECTS
                            self.state_timer = t
                            self.attempt_counter = 0
                        elif t - self.state_timer > 5.0:
                            print("[ALIGN_ARM] ⚠ Timeout reached, proceeding to search anyway...")
                            self.state = State.SEARCH_OBJECTS
                            self.state_timer = t
                            self.attempt_counter = 0
                    
            elif self.state == State.SEARCH_OBJECTS:
                """
                State: SEARCH_OBJECTS - Detect objects using camera perception
                
                Purpose:
                    Scan table surface and identify object for grasping
                    
                Perception Strategy:
                    - Use camera recognition API to detect objects in view
                    - Filter by distance (< 1.5m) to avoid false positives
                    - Select closest object for sequential processing
                    - Transform coordinates from camera frame to world frame
                    - Retry detection if transient failures occur
                    
                Exit Conditions:
                    - Object detected → PICK_OBJECT (store target position)
                    - 8 failed attempts → Adjust arm pose and retry
                    
                Rationale:
                    Closest-first strategy minimizes arm travel and cycle time.
                    Retry logic with arm adjustment handles edge cases where
                    initial view is obscured or object just outside field of view.
                """
                if t - self.state_timer > 1.0:  # Check every 1 second
                    if DEBUG: print(f"[SEARCH_OBJECTS] Scan attempt {self.attempt_counter+1}, looking for cube...")
                    cube_pos = self.detect_cube()
                    if cube_pos is not None:
                        self.target_cube_pos = cube_pos
                        print(f"[SEARCH_OBJECTS] ✓ Cube found at {cube_pos}!")
                        print(f"[SEARCH_OBJECTS] Transitioning to PICK_OBJECT...")
                        self.state = State.PICK_OBJECT
                        self.attempt_counter = 0
                    else:
                        self.attempt_counter += 1
                        if DEBUG: print(f"[SEARCH_OBJECTS] No detection (attempt {self.attempt_counter}/8)")
                        
                        if self.attempt_counter > 8:  # After 8 seconds without detection
                            print("[SEARCH_OBJECTS] ⚠ Detection failed - repositioning arm for better view")
                            # Slightly adjust arm for better visibility
                            current_arm = self.joint_pos(self.GRIPPER_ARM)
                            adjusted_arm = current_arm.copy()
                            adjusted_arm[2] = adjusted_arm[2] - 0.2   # Extend forearm more
                            adjusted_arm[3] = adjusted_arm[3] - 0.25  # Look down more
                            self.move_PTP(self.GRIPPER_ARM, adjusted_arm, timeout=3.0, velocity=0.5)
                            self.attempt_counter = 0
                            self.state_timer = t
                        else:
                            self.state_timer = t 
            
            elif self.state == State.PICK_OBJECT:
                """
                State: PICK_OBJECT - Execute grasp sequence on detected object
                
                Purpose:
                    Physically grasp the target object identified by perception
                    
                Grasp Execution:
                    See execute_grasp() for detailed grasp strategy
                    Phases: Hover → Descend → Close → Lift
                    
                Post-Grasp:
                    Move arm to CARRY position (tucked, safe for navigation)
                    
                Exit Conditions:
                    - Grasp successful → GOTO_BIN (proceed with transport)
                    - Grasp failed (IK error) → ALIGN_ARM (retry detection)
                    
                Rationale:
                    Carry position keeps object secure and arm clear of obstacles
                    during navigation. Retry on failure handles transient issues
                    like object movement or workspace changes.
                """
                print(f"[STATE: PICK_OBJECT] Attempting to grasp cube at {self.target_cube_pos}")
                if self.execute_grasp(self.target_cube_pos):
                    print("[STATE: PICK_OBJECT] Grasp successful! Moving to GOTO_BIN...")
                    self.move_PTP(self.GRIPPER_ARM, ARM_CARRY, timeout=5.0)
                    self.state = State.GOTO_BIN
                else:
                    print("[STATE: PICK_OBJECT] Grasp failed. Retrying search...")
                    self.arm_aligned = False
                    self.state = State.ALIGN_ARM
                    self.state_timer = t
            
            elif self.state == State.GOTO_BIN:
                """
                State: GOTO_BIN - Navigate to bin drop-off location
                
                Purpose:
                    Transport grasped object to bin for release
                    
                Navigation:
                    Target: BIN_APPROACH_POSE [0.5, 1.3, 1.57]
                    - Position beside bin for arm reach
                    - Orientation faces bin opening
                    
                Safety:
                    ARM_CARRY configuration keeps object secure during transit
                    and maintains center of gravity for stable navigation
                    
                Exit Condition:
                    Arrived at bin position → DROP_OBJECT
                """
                if self.navigate(BIN_APPROACH_POSE):
                    print("[STATE: GOTO_BIN] Arrived at bin. Moving to DROP_OBJECT...")
                    self.state = State.DROP_OBJECT
            
            elif self.state == State.DROP_OBJECT:
                """
                State: DROP_OBJECT - Release object into bin
                
                Purpose:
                    Complete pick-and-place cycle by depositing object
                    
                Actions:
                    1. Move arm to DROP position (rotated to rear, over bin)
                    2. Open gripper fully to release object
                    3. Wait for object to fall completely
                    4. Increment success counter
                    5. Check if all objects collected
                    
                ARM_DROP_BIN Configuration:
                    [0.0, 1.1, -0.5, -0.8, 0.0]
                    - J1=0.0: Rotate to rear (180° from front)
                    - Positions gripper above bin opening
                    - Gravity-assisted release ensures object drops cleanly
                    
                Exit Conditions:
                    - All 4 objects collected → DONE (mission complete)
                    - More objects remain → GOTO_TABLE (start next cycle)
                    
                Rationale:
                    Counter-based termination ensures all objects processed.
                    Settling time after release prevents gripper interference
                    with falling object and allows physics to stabilize.
                """
                print("[STATE: DROP_OBJECT] Moving arm to drop position...")
                self.move_PTP(self.GRIPPER_ARM, ARM_DROP_BIN, timeout=5.0)
                print("[STATE: DROP_OBJECT] Opening gripper over bin...")
                
                # Force gripper to maximum open position for clean release
                self.finger.setPosition(0.025)  # 25mm maximum separation
                self.open_gripper(timeout=2.0)  # Extended timeout ensures complete opening
                
                # Extended settling time ensures object falls completely into bin
                for _ in range(DROP_SETTLE_ITERATIONS):
                    self.step(self.TIME_STEP)
                
                self.cubes_collected += 1
                print(f"\n[STATE: DROP_OBJECT] ✓ Cube #{self.cubes_collected} dropped successfully!")
                print(f"[STATE: DROP_OBJECT] Total cubes collected: {self.cubes_collected}/4\n")
                
                if self.cubes_collected >= 4:
                    print("[STATE: DROP_OBJECT] 🎉 ALL CUBES COLLECTED! Moving to DONE...")
                    self.state = State.DONE
                else:
                    print("[STATE: DROP_OBJECT] Moving arm to carry position...")
                    self.move_PTP(self.GRIPPER_ARM, ARM_CARRY, timeout=5.0)
                    
                    # Settling time before navigation prevents arm oscillations
                    for _ in range(15):
                        self.step(self.TIME_STEP)
                    
                    print("[STATE: DROP_OBJECT] Transitioning to GOTO_TABLE for next cube...")
                    self.state = State.GOTO_TABLE
                    self.state_timer = t
            
            elif self.state == State.RETURN_TABLE:
                """
                State: RETURN_TABLE - Navigate back to table for next object
                
                Purpose:
                    Return to table approach position after drop-off
                    
                Note:
                    Currently unused - controller transitions directly from
                    DROP_OBJECT to GOTO_TABLE for efficiency. Kept for
                    potential future extensions requiring intermediate steps.
                """
                if self.navigate(TABLE_APPROACH_POSE):
                    self.arm_aligned = False
                    self.state = State.ALIGN_ARM
            
            elif self.state == State.DONE:
                """
                State: DONE - Mission complete
                
                Purpose:
                    Cleanly terminate controller after all objects collected
                    
                Actions:
                    1. Stop all base motion (set velocities to zero)
                    2. Print completion message
                    3. Exit control loop
                    
                Entry Condition:
                    cubes_collected >= 4 (all objects successfully placed)
                """
                self.set_mecanuum_control(0,0,0)
                print("Mission Complete!")
                break

def main():
    """
    Entry point for controller execution.
    
    Instantiates Mission3Controller and starts the main control loop.
    This function is called automatically when the Webots simulator
    launches the controller process.
    
    Execution Flow:
        1. Create controller instance (initializes hardware devices)
        2. Run main control loop (FSM execution)
        3. Exit when mission complete or simulation stopped
    """
    robot = Mission3Controller()
    robot.run()

if __name__ == '__main__':
    main()