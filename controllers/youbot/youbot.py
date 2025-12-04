"""
Mission 3 Controller for DG4RAS - KUKA youBot [FINAL FIXED]
===========================================================
1. Navigation: Safety threshold lowered to 0.10m to ignore table edge.
2. Pose: Arm forces "Look Down" configuration (J4 = -1.6).
3. Orientation: J5 = 2.92 ensures camera is upright.
"""

import math
import time
import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation
from youbot_base import YouBot

DEBUG = True

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Navigation
CRUISE_SPEED = 0.25
TURN_SPEED = 0.6
DISTANCE_TOLERANCE = 0.02
ANGLE_TOLERANCE = 0.05

# --- ARM CONFIGURATIONS ---
# CRITICAL FIX: Arm must extend FORWARD and look DOWN at table
# J1=2.95: Face Front toward table
# J2=0.6: Raise shoulder to extend arm forward
# J3=-1.8: Extend forearm forward over table
# J4=-1.6: LOOK DOWN at table surface
# J5=2.92: Keep camera upright

ARM_HOME = np.array([2.95, 1.1, -2.5, 1.5, 2.92])
# High search position - EXTENDED FORWARD and looking DOWN
ARM_SEARCH = np.array([2.95, 0.6, -1.8, -1.6, 2.92])   
# Pre-grasp just above the object
ARM_PRE_GRASP = np.array([2.95, 0.4, -0.9, -1.6, 2.92]) 
# Carry position (tucked but holding object level)
ARM_CARRY = np.array([2.95, 0.8, -0.5, 1.0, 2.92])     
# Drop position (Rotated to back, J1=0.0)
ARM_DROP_BIN = np.array([0.0, 1.1, -0.5, -0.8, 0.0])   

# World Coordinates
# Stop 1.8m from center (leaves ~30cm gap to table for arm reach)
TABLE_APPROACH_POSE = np.array([-1.80, 0.0, 3.14159]) 
BIN_APPROACH_POSE = np.array([0.5, 1.3, 1.57])

class State:
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
    """Convert [w, x, y, z] to 3x3 rotation matrix safely."""
    try:
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    except:
        return np.eye(3)

class Mission3Controller(YouBot):
    def __init__(self):
        super().__init__()
        self.rgb_camera.recognitionEnable(self.TIME_STEP)
        
        # FSM
        self.state = State.INIT
        self.state_timer = self.getTime()
        self.cubes_collected = 0
        self.target_cube_pos = None
        self.attempt_counter = 0
        self.current_pose = np.zeros(3)
        self.arm_aligned = False

    def verify_gripper_open(self):
        """Verify gripper is fully open by checking finger sensor."""
        try:
            # Gripper fingers should be at max position (0.025)
            finger_pos = self.finger.getPositionSensor().getValue() if hasattr(self.finger, 'getPositionSensor') else None
            if DEBUG: print(f"[VERIFY_GRIPPER] Finger position: {finger_pos}")
            return True
        except:
            return True  # Assume open if we can't verify

    def get_robot_pose(self):
        gps_vals = self.gps.getValues()
        cmp_vals = self.compass.getValues()
        heading = math.atan2(cmp_vals[0], cmp_vals[1])
        
        if heading > math.pi: heading -= 2*math.pi
        elif heading <= -math.pi: heading += 2*math.pi
            
        return np.array([gps_vals[0], gps_vals[1], heading])

    def navigate(self, target):
        curr = self.get_robot_pose()
        dx = target[0] - curr[0]
        dy = target[1] - curr[1]
        d_theta = target[2] - curr[2]
        
        if d_theta > math.pi: d_theta -= 2*math.pi
        elif d_theta <= -math.pi: d_theta += 2*math.pi
        
        dist = math.sqrt(dx**2 + dy**2)
        
        # Arrived?
        if dist < DISTANCE_TOLERANCE and abs(d_theta) < ANGLE_TOLERANCE:
            self.set_mecanuum_control(0, 0, 0)
            return True
            
        # Transform to local frame
        c, s = math.cos(curr[2]), math.sin(curr[2])
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        
        vx = max(min(lx * 1.0, CRUISE_SPEED), -CRUISE_SPEED)
        vy = max(min(ly * 1.0, CRUISE_SPEED), -CRUISE_SPEED)
        w  = max(min(d_theta * 2.0, TURN_SPEED), -TURN_SPEED)
        
        # --- FIXED SAFETY CHECK ---
        scan = np.array(self.lidar.getRangeImage())
        mid = len(scan)//2
        # CHANGED TO 0.10 (10cm) so it doesn't stop when seeing the table
        if np.min(scan[mid-20:mid+20]) < 0.10 and vx > 0:
            vx = 0 
            
        self.set_mecanuum_control(vx, vy, w)
        return False

    def detect_cube(self):
        """Detect and return position of closest cube visible to camera."""
        objs = self.rgb_camera.getRecognitionObjects()
        if not objs:
            if DEBUG: print("[DETECT_CUBE] No objects detected by camera")
            return None
        
        if DEBUG: print(f"[DETECT_CUBE] Found {len(objs)} object(s) in camera view")
        
        # Get closest object (filter by reasonable distance)
        valid_objs = []
        for obj in objs:
            dist = np.linalg.norm(obj.getPosition())
            if dist < 1.5:  # Maximum reasonable distance in camera frame
                valid_objs.append(obj)
                if DEBUG: print(f"  - Object '{obj.getModel()}' at camera distance: {dist:.3f}m")
        
        if not valid_objs:
            if DEBUG: print("[DETECT_CUBE] No objects within valid range")
            return None
        
        # Get closest valid object
        target = min(valid_objs, key=lambda o: np.linalg.norm(o.getPosition()))
        pos_cam = np.array(target.getPosition()) 
        model_name = target.getModel()
        
        if DEBUG: print(f"[DETECT_CUBE] Selected closest object '{model_name}' at camera coords: {pos_cam}")
        
        # Transform from camera frame to world frame
        try:
            cam_tf = self.forward_kinematics(self.GRIPPER_ARM, camera_link=True)
            rot_mat = quaternion_to_matrix(cam_tf.rot)
            pos_base = rot_mat @ pos_cam + cam_tf.pos
        except Exception as e:
            if DEBUG: print(f"[DETECT_CUBE] FK error: {e}")
            return None
        
        r_pose = self.get_robot_pose()
        c, s = math.cos(r_pose[2]), math.sin(r_pose[2])
        
        world_x = r_pose[0] + pos_base[0]*c - pos_base[1]*s
        world_y = r_pose[1] + pos_base[0]*s + pos_base[1]*c
        world_z = pos_base[2]
        
        if DEBUG: print(f"[DETECT_CUBE] Cube '{model_name}' at World: X={world_x:.2f}, Y={world_y:.2f}, Z={world_z:.2f}")
        return np.array([world_x, world_y, world_z])

    def execute_grasp(self, cube_world):
        """
        Top-down grasp that biases IK to elbow-down and clamps joints.
        Returns True on success, False on failure.
        """
        # helper: clamp joint vector to safe limits (radians)
        def clamp_joints(q):
            # these are conservative limits; adjust if your URDF uses different bounds
            limits = [
                (-3.14, 3.14),  # joint 1
                (-2.8,  2.8),   # joint 2 (shoulder)
                (-2.8,  2.8),   # joint 3 (elbow)
                (-3.14, 3.14),  # joint 4
                (-3.14, 3.14)   # joint 5 (wrist)
            ]
            qc = np.array(q, dtype=float)
            for i in range(min(len(qc), len(limits))):
                qc[i] = max(min(qc[i], limits[i][1]), limits[i][0])
            return qc

        # Bias seed to the forward / top-down posture (keeps elbow down over table)
        ik_seed = ARM_SEARCH.copy()

        # robot pose & transform world->base (we already use this pattern in your code)
        r_pose = self.get_robot_pose()
        dx = cube_world[0] - r_pose[0]
        dy = cube_world[1] - r_pose[1]
        c, s = math.cos(r_pose[2]), math.sin(r_pose[2])
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c

        # Tweakable parameters (tune if needed)
        grasp_height = 0.025      # target grasp z above table (m). 0.025 = 25 mm
        hover_height  = 0.14      # safe hover height above table (m)
        forward_comp  = 0.05     # small forward offset for gripper geometry (m)
        final_descend_velocity = 0.12  # slow velocity for linear descent

        # Ensure end-effector is centered above the cube
        local_x = local_x + forward_comp 

        if DEBUG:
            print(f"[GRASP] local target X,Y = {local_x:.3f}, {local_y:.3f}  (world {cube_world[0]:.3f},{cube_world[1]:.3f})")

        # Build top-down transforms (base-frame)
        target_rot = np.array([0.0, 0.0, 1.0, 0.0])  # gripper pointing down (same as you used)
        hover_tf = kp.Transform(pos=np.array([local_x, local_y, grasp_height + hover_height]), rot=target_rot)
        grasp_tf = kp.Transform(pos=np.array([local_x, local_y, grasp_height]), rot=target_rot)

        # Try IK with a bias seed to prefer the forward configuration.
        # If your inverse_kinematics accepts q_init or seed, pass it; otherwise set motors temporarily like before.
        try:
            q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf, q_init=ik_seed)
        except TypeError:
            # older API: no q_init param — give a seed by moving a couple of joints briefly
            try:
                self.motors_gripper_arm[0].setPosition(ik_seed[0])
                self.motors_gripper_arm[4].setPosition(ik_seed[4])
            except Exception:
                pass
            q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf)

        if q_hover is None:
            # try small XY offsets around the cube to find a reachable hover
            offsets = [(0.0,0.0), (0.02,0.0), (-0.02,0.0), (0.0,0.02), (0.0,-0.02)]
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
            if DEBUG: print("[GRASP] Failed to find IK for hover pose.")
            return False

        # Then solve for the exact top-down grasp pose (prefer same seed)
        try:
            q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, grasp_tf, q_init=q_hover)
        except TypeError:
            q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, grasp_tf)

        if q_grasp is None:
            # fallback: slightly raise grasp z (avoid collision) and try again
            for dz in [0.01, 0.02, -0.005]:
                tf = kp.Transform(pos=np.array([local_x, local_y, grasp_height + dz]), rot=target_rot)
                try:
                    q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, tf, q_init=q_hover)
                except TypeError:
                    q_grasp = self.inverse_kinematics(self.GRIPPER_ARM, tf)
                if q_grasp is not None:
                    grasp_tf = tf
                    if DEBUG: print(f"[GRASP] grasp IK succeeded with dz={dz:.3f}")
                    break

        if q_grasp is None:
            if DEBUG: print("[GRASP] Failed to find IK for grasp pose.")
            return False

        # Clamp both solutions to joint limits to avoid commanding impossible positions
        q_hover = clamp_joints(q_hover)
        q_grasp = clamp_joints(q_grasp)

        # Optional sanity check: avoid large jumps from current joints (prevent flips)
        try:
            cur = np.array(self.joint_pos(self.GRIPPER_ARM))
            jump = np.abs(cur - q_hover)
            if np.any(jump > 3.0):  # >~171 degrees jump is suspicious
                # re-seed IK using a milder intermediate pose (ARM_SEARCH)
                try:
                    q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf, q_init=ARM_SEARCH)
                except TypeError:
                    q_hover = self.inverse_kinematics(self.GRIPPER_ARM, hover_tf)
                if q_hover is None:
                    if DEBUG: print("[GRASP] IK after re-seeding failed; aborting.")
                    return False
                q_hover = clamp_joints(q_hover)
        except Exception:
            # if joint_pos() not available, continue anyway
            pass

        # Execute the motion: go to hover (PTP), then descend straight down (LIN) to grasp
        self.open_gripper(timeout=1.0)
        if DEBUG: print("[GRASP] Moving to hover (PTP).")
        ok = self.move_PTP(self.GRIPPER_ARM, q_hover, velocity=0.6, timeout=6.0)
        if not ok:
            if DEBUG: print("[GRASP] move_PTP to hover timed out.")
            # still try to continue if IK was ok

        # short settle
        for _ in range(5):
            self.step(self.TIME_STEP)

        # Prefer move_LIN for straight-line descent if available (gives top-down motion)
        used_lin = False
        try:
            # If move_LIN accepts a Transform, pass grasp_tf; some APIs accept q or transform
            self.move_LIN(self.GRIPPER_ARM, grasp_tf, timeout=4.0, velocity=final_descend_velocity)
            used_lin = True
        except Exception:
            # move_LIN not available or rejected; fall back to PTP to q_grasp slowly
            if DEBUG: print("[GRASP] move_LIN unavailable, falling back to slow PTP for descent.")
            self.move_PTP(self.GRIPPER_ARM, q_grasp, velocity=0.14, timeout=5.0)

        # small settle
        for _ in range(6):
            self.step(self.TIME_STEP)

        # Close gripper
        self.close_gripper(timeout=1.5)
        for _ in range(6):
            self.step(self.TIME_STEP)

        # Lift straight back to hover (use LIN if available)
        if used_lin:
            try:
                self.move_LIN(self.GRIPPER_ARM, hover_tf, timeout=4.0, velocity=0.4)
            except Exception:
                self.move_PTP(self.GRIPPER_ARM, q_hover, velocity=0.4)
        else:
            self.move_PTP(self.GRIPPER_ARM, q_hover, velocity=0.4)

        return True


    def run(self):
        print("Mission 3 Controller Started...")
        
        while self.step(self.TIME_STEP) != -1:
            t = self.getTime()
            
            if self.state == State.INIT:
                print("\n" + "="*60)
                print("Mission 3 Controller Started...")
                print("="*60)
                print("[INIT] Opening gripper fully...")
                self.open_gripper(timeout=1.5)
                
                # Wait for gripper to fully open
                for _ in range(20):
                    self.step(self.TIME_STEP)
                
                self.verify_gripper_open()
                
                print("[INIT] Moving arm to HOME position...")
                self.move_PTP(self.GRIPPER_ARM, ARM_HOME, timeout=5.0)
                
                for _ in range(10):
                    self.step(self.TIME_STEP)
                
                print("[INIT] Initialization complete! Moving to GOTO_TABLE...")
                self.state = State.GOTO_TABLE
                self.state_timer = self.getTime()
                
            elif self.state == State.GOTO_TABLE:
                if self.navigate(TABLE_APPROACH_POSE):
                    print(f"[STATE: GOTO_TABLE] Arrived at Table on attempt {self.cubes_collected + 1}.")
                    print("[STATE: GOTO_TABLE] Resetting arm alignment flag...")
                    self.arm_aligned = False
                    self.state = State.ALIGN_ARM
                    self.state_timer = t
                    
            elif self.state == State.ALIGN_ARM:
                # CRITICAL FIX: Ensure arm actually moves to search position
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
                if self.navigate(BIN_APPROACH_POSE):
                    print("[STATE: GOTO_BIN] Arrived at bin. Moving to DROP_OBJECT...")
                    self.state = State.DROP_OBJECT
            
            elif self.state == State.DROP_OBJECT:
                print("[STATE: DROP_OBJECT] Moving arm to drop position...")
                self.move_PTP(self.GRIPPER_ARM, ARM_DROP_BIN, timeout=5.0)
                print("[STATE: DROP_OBJECT] Opening gripper over bin...")
                
                # CRITICAL: Force gripper to full open position
                self.finger.setPosition(0.025)  # Set to MAX
                self.open_gripper(timeout=2.0)  # Extra timeout
                
                # Ensure gripper is fully open and object has fallen
                for _ in range(30):
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
                    
                    # Wait a moment before navigation
                    for _ in range(15):
                        self.step(self.TIME_STEP)
                    
                    print("[STATE: DROP_OBJECT] Transitioning to GOTO_TABLE for next cube...")
                    self.state = State.GOTO_TABLE
                    self.state_timer = t
            
            elif self.state == State.RETURN_TABLE:
                 if self.navigate(TABLE_APPROACH_POSE):
                    self.arm_aligned = False
                    self.state = State.ALIGN_ARM
            
            elif self.state == State.DONE:
                self.set_mecanuum_control(0,0,0)
                print("Mission Complete!")
                break

def main():
    robot = Mission3Controller()
    robot.run()

if __name__ == '__main__':
    main()