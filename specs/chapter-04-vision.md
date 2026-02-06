# Specification: Chapter 4 - Computer Vision with RealSense

## Learning Objectives
- Install and configure the Intel RealSense SDK for ROS 2
- Understand RGB-D (Depth) data and its importance for robotic perception
- Visualize camera data streams in Rviz
- Distinguish between 2D (Pixel) and 3D (Point Clouds) representation
- Learn how a robot measures distance using stereo vision or structured light
- Deploy and launch RealSense camera drivers on physical or simulated robots

## Core Concepts
1. **Difference between 2D (Pixels) and 3D (Point Clouds)**:
   - 2D Images consist of pixels with intensity/color values arranged in a grid, representing what the camera sees from a single viewpoint
   - 3D Point clouds represent spatial coordinates (x, y, z) of points in space, providing geometric information about object surfaces
   - Point clouds enable robots to understand spatial relationships, depths, and volumes

2. **How a robot measures distance**:
   - Stereo vision: Uses two cameras to calculate depth based on parallax differences
   - Structured light: Projects known patterns and measures distortions to infer depth
   - Time-of-flight: Measures the time taken for light to return from objects (if supported by sensor)

## Hands-On Implementation
1. **Installation**:
   - Install the RealSense2 camera package: `sudo apt install ros-humble-realsense2-camera`
   - Ensure udev rules are properly set up for device access

2. **Launching the camera node**:
   - Run the RealSense camera driver using the launch file
   - Verify the camera is publishing data to ROS topics

3. **Visualization**:
   - Use Rviz to visualize the camera feed and depth data
   - Configure Rviz displays to show both RGB and depth streams

## Diagrams
- **Data Flow Diagram**: Visualizing the flow from Camera -> /camera/color/image_raw -> Rviz
- **Conceptual Diagram**: Illustrating the difference between 2D images and 3D point clouds
- **Hardware Diagram**: Showing RealSense camera mounting on robot platform

## Essential Commands
- `ros2 launch realsense2_camera rs_launch.py` - Launch the RealSense camera driver
- `ros2 topic list | grep camera` - List all camera-related topics
- `ros2 topic echo /camera/color/image_raw` - View raw image data
- `ros2 run rviz2 rviz2` - Launch Rviz for visualization
- `rs-enumerate-devices` - Check connected RealSense devices
- `realsense-viewer` - Open RealSense viewer application for debugging

## Evaluation Criteria
- Successfully launch RealSense camera driver
- Visualize RGB and depth data in Rviz
- Understand the differences between 2D and 3D representations
- Troubleshoot common connection and calibration issues