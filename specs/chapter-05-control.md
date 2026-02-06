# Specification: Chapter 5 - Arm Control & Motion Planning

## Learning Objectives
- Understand URDF (Unified Robot Description Format) and its role in representing robot structure
- Master the fundamentals of MoveIt 2 for motion planning
- Learn Trajectory Planning concepts and implementation
- Visualize robot models in Rviz
- Use MoveIt's MotionPlanning plugin for interactive arm control
- Implement collision-free path planning between waypoints

## Core Concepts
1. **What is Motion Planning?**
   - Motion planning is the computational problem of finding a path or trajectory for a robot to move from an initial configuration to a goal configuration while avoiding obstacles
   - The process involves determining a sequence of valid configurations that connect the start and goal states
   - Motion planning algorithms must account for robot kinematics, dynamics, obstacles, and environmental constraints
   - Key challenges include handling high-dimensional configuration spaces and real-time replanning

2. **URDF (Unified Robot Description Format)**
   - URDF is an XML-based format used in ROS to describe the physical properties of a robot
   - It serves as the digital twin of the robot structure, defining links, joints, inertial properties, visual elements, and collision properties
   - URDF files contain information about robot geometry, kinematics, and dynamics
   - URDF enables simulation, visualization, and control of robots in ROS environments

## Hands-On Implementation
1. **Installation**:
   - Install MoveIt 2 packages: `sudo apt install ros-humble-moveit`
   - Verify installation and dependencies

2. **Visualization**:
   - Load and visualize the Robot Model (URDF) in Rviz
   - Set up the robot state publisher to broadcast TF transforms
   - Configure Rviz displays to show the robot model

3. **Motion Planning**:
   - Launch the MotionPlanning plugin in Rviz
   - Use the interactive markers to drag and move the arm virtually
   - Plan trajectories and execute them in simulation
   - Test obstacle avoidance with virtual objects in the environment

## Diagrams
- **MoveIt Pipeline Diagram**: Visualizing the flow from User Request -> MoveGroup Node -> Controllers -> Robot Hardware
- **URDF Structure Diagram**: Showing links and joints of a robotic arm
- **Motion Planning Workflow**: Illustrating the process from goal specification to trajectory execution

## Essential Commands
- `ros2 launch moveit_setup_assistant setup_assistant.launch.py` - Launch the MoveIt Setup Assistant for robot configuration
- `ros2 launch <robot_name>_moveit_config move_group.launch.py` - Launch the MoveGroup node for motion planning
- `ros2 run rviz2 rviz2` - Launch Rviz for visualization
- `ros2 launch <robot_name>_moveit_config demo.launch.py` - Launch MoveIt demo for testing
- `ros2 param list` - Check ROS parameters
- `ros2 action list` - List active actions including motion planning services

## Evaluation Criteria
- Successfully install MoveIt 2 packages
- Visualize a robot model in Rviz using URDF
- Use MotionPlanning plugin to plan and execute arm movements
- Demonstrate collision-free path planning
- Understand the relationship between URDF, MoveIt, and robot control