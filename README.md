# KUKA youBot Pick-and-Place Controller

Autonomous pick-and-place controller for KUKA youBot robot using Webots simulation environment.

## Overview

This project implements a complete autonomous controller for a KUKA youBot mobile manipulator to perform pick-and-place operations. The robot detects objects (cubes) on a table using camera-based perception, grasps them sequentially using a top-down approach strategy, and places them into a bin.

## Features

### Perception System
- **Object Detection**: RGB camera with built-in object recognition
- **Coordinate Transformation**: Forward kinematics for camera-to-world frame conversion
- **Object Selection**: Closest-first strategy for efficiency
- **False Positive Filtering**: Distance-based filtering (< 1.5m threshold)

### Control System
- **Inverse Kinematics**: Grasp pose planning with IK seed biasing
- **Forward Kinematics**: End-effector pose computation
- **Motion Primitives**:
  - Point-to-Point (PTP): Fast joint-space movement
  - Linear (LIN): Cartesian straight-line paths
- **Base Navigation**: Omnidirectional mecanum wheel control with proportional feedback

### Behavior Coordination
- **Finite State Machine**: 8-state FSM orchestrates task execution
- **State Flow**: INIT → GOTO_TABLE → ALIGN_ARM → SEARCH_OBJECTS → PICK_OBJECT → GOTO_BIN → DROP_OBJECT → (loop) → DONE
- **Error Handling**: Retry logic for transient failures
- **Progress Tracking**: Sequential processing of all objects

### Manipulation Strategy
- **Top-Down Grasp**: Vertical approach for stable object acquisition
- **Four-Phase Execution**:
  1. **Hover**: Move to safe height above object (14cm clearance)
  2. **Descend**: Lower straight down to grasp height (2.5cm)
  3. **Close**: Engage gripper fingers
  4. **Lift**: Raise back to hover position
- **Safety Features**:
  - Joint clamping prevents self-collision
  - IK sanity checks detect configuration flips
  - LiDAR-based obstacle avoidance

## Project Structure

```
robotics/
├── controllers/
│   └── youbot/
│       ├── youbot.py           # Main pick-and-place controller
│       └── youbot_base.py      # Base class with hardware interface
├── worlds/
│   └── mission3.wbt            # Webots simulation world
├── resources/
│   └── youbot.urdf             # Robot kinematics model
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

1. **Webots R2023b or later**: Download from [cyberbotics.com](https://cyberbotics.com)
2. **Python 3.8+**: Included with Webots or install separately

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DheVam/robotics.git
cd robotics
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Note: The `controller` module is provided by Webots and doesn't need separate installation.

## Usage

### Running in Webots

1. Open Webots
2. Load the world file: `File → Open World → worlds/mission3.wbt`
3. Click the Play button to start the simulation
4. The controller will automatically:
   - Initialize hardware
   - Navigate to the table
   - Detect and grasp objects sequentially
   - Transport objects to the bin
   - Repeat until all 4 objects are collected

### Debug Mode

Enable verbose logging by setting `DEBUG = True` in `controllers/youbot/youbot.py` (default).

Output includes:
- State transitions
- Object detection results
- Coordinate transformations
- Grasp execution phases
- Navigation status

## Configuration

### Adjustable Parameters

Edit constants in `youbot.py` to tune behavior:

```python
# Navigation
CRUISE_SPEED = 0.25        # Linear velocity [m/s]
TURN_SPEED = 0.6           # Angular velocity [rad/s]
DISTANCE_TOLERANCE = 0.02  # Position accuracy [m]
ANGLE_TOLERANCE = 0.05     # Heading accuracy [rad]

# Obstacle Avoidance
OBSTACLE_THRESHOLD = 0.10  # Minimum safe distance [m]

# Grasp Execution
grasp_height = 0.025       # Final grasp height [m]
hover_height = 0.14        # Approach clearance [m]
forward_comp = 0.05        # Gripper offset [m]
```

### Arm Configurations

Pre-defined joint configurations (5-DOF arm):

- **ARM_HOME**: Safe stowed position
- **ARM_SEARCH**: Extended with camera looking down
- **ARM_CARRY**: Tucked for safe transport
- **ARM_DROP_BIN**: Positioned over bin

### World Waypoints

Fixed navigation targets [x, y, theta]:

- **TABLE_APPROACH_POSE**: [-1.80, 0.0, π] - In front of table
- **BIN_APPROACH_POSE**: [0.5, 1.3, π/2] - Beside bin

## Design Rationale

### Why Top-Down Grasp?
- Gripper fingers aligned with gravity provide stable grip
- Vertical path minimizes collision risk
- Consistent approach angle simplifies planning
- Maintains camera view throughout sequence

### Why Closest-First Selection?
- Minimizes arm travel distance
- Reduces total cycle time
- Simplifies coordination (no path planning between objects)

### Why Finite State Machine?
- Clear separation of concerns
- Easy to understand and debug
- Robust error handling with retry logic
- Deterministic behavior

## Performance

Expected performance metrics:
- **Detection Rate**: >95% (with proper lighting and camera positioning)
- **Grasp Success**: >90% (top-down approach is highly reliable)
- **Cycle Time**: ~30-40 seconds per object
- **Total Mission Time**: ~2-3 minutes for 4 objects

## Troubleshooting

### Object Not Detected
- Check camera is pointing at table (ARM_SEARCH configuration)
- Verify objects are within 1.5m of camera
- Enable DEBUG mode to see detection attempts

### Grasp Failure
- Object may be too close to workspace boundary
- IK solver unable to find solution
- Controller will retry after repositioning arm

### Navigation Issues
- Check LiDAR is not blocked
- Verify waypoint coordinates match world layout
- Adjust OBSTACLE_THRESHOLD if robot stops too early

## Future Enhancements

Possible improvements:
- [ ] Collision checking in trajectory planning
- [ ] Adaptive grasp height based on object size
- [ ] Parallel detection during navigation (reduce cycle time)
- [ ] Multi-object grasping (use tray arm)
- [ ] Dynamic obstacle avoidance during manipulation

## References

- [Webots Documentation](https://cyberbotics.com/doc/guide/index)
- [KUKA youBot Specifications](https://www.kuka.com/en-de/products/mobility/mobile-robots/kuka-youbot)
- [kinpy Library](https://pypi.org/project/kinpy/)

## License

This project is part of the DG4RAS course assessment.

## Authors

- Controller Implementation: Comprehensive autonomous pick-and-place system
- Documentation: Design rationale and implementation details

---

For questions or issues, please refer to the code comments or contact the course instructor.
