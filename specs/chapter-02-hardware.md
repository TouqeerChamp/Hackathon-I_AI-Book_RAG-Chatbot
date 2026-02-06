# Chapter 2 Specification: Hardware Lab Setup

## Overview
This chapter focuses on setting up the physical infrastructure required for working with humanoid robotics and embodied AI. It covers the selection of appropriate hardware components, their specifications, and how to establish a functional lab environment for development and experimentation.

## Learning Objectives
By the end of this chapter, readers should be able to:
- Configure a Physical AI Workstation suitable for robotics applications
- Set up a Robot Brain using appropriate edge computing platforms
- Identify and select appropriate hardware components for humanoid robotics
- Establish networking connections between robotic systems and development computers

## Hardware Bill of Materials (BOM)

### Edge Computer
- **Primary Option**: NVIDIA Jetson Orin Nano
  - 128 CUDA cores
  - 2GB or 4GB LPDDR5 memory
  - 15W power consumption
  - ARM64 architecture
- **Alternative Option**: NVIDIA Jetson AGX Orin
  - 2048 CUDA cores
  - 32GB LPDDR5 memory
  - Up to 60W power consumption
  - ARM64 architecture

### Vision System
- **RealSense Depth Camera D435i**
  - RGB resolution: 1920x1080 at 30 FPS
  - Depth resolution: 1280x720 at 90 FPS
  - Integrated IMU for motion tracking
  - USB 3.0 connectivity
  - Stereo depth technology

### Robot Arms
- **Primary Option**: WidowX 250 6-DOF Robotic Arm
  - 6 degrees of freedom for dexterous manipulation
  - Reach: 250mm
  - Payload capacity: 100g at full reach
  - Compatible with Robotis Dynamixel servos
  - Open-source hardware design
- **Alternative Options**: Equivalent 6-DOF arm with similar specifications

### Simulation PC
- **Minimum GPU Requirement**: NVIDIA RTX 3060 (12GB VRAM)
- **Recommended GPU**: NVIDIA RTX 4070 or higher
- **CPU**: Modern multi-core processor (Intel i7 or AMD Ryzen 7)
- **RAM**: 32GB or more
- **OS**: Ubuntu 22.04 LTS (compatible with ROS 2 Humble)
- **Storage**: SSD with 500GB+ free space
- **Purpose**: Running Gazebo simulation, Isaac Sim, and training AI models

## Key Concepts

### ARM64 vs x86 Architecture
- Differences between ARM64 and x86 instruction sets
- Advantages of ARM64 for embedded robotics applications (power efficiency, thermal management)
- Compatibility considerations for developing cross-platform software
- Docker containerization strategies for ARM64 devices

### CUDA Cores
- Understanding CUDA cores and their role in parallel processing
- Importance of CUDA cores for running neural networks on GPUs
- CUDA compute capabilities and compatibility with robotics software stacks
- Balancing cost versus performance when selecting GPUs for robotics

### Tensor Cores
- Purpose and benefits of Tensor Cores for AI inference
- How Tensor Cores accelerate deep learning models
- Requirements for utilizing Tensor Cores in robotics applications
- Considerations for selecting NVIDIA hardware with Tensor Core capabilities

## Wiring & Networking

### SSH Setup
- Configuring SSH server on Jetson Orin devices
- Generating SSH key pairs for secure connections
- Setting up static IP addresses for reliable robot connections
- Troubleshooting common SSH connection issues

### Network Connection Between Devices
- **Ethernet Connection**:
  - Direct cable connection between laptop and robot
  - Static IP configuration for predictable connections
  - Bandwidth considerations for sensor data transmission
- **WiFi Connection**:
  - Wireless setup for mobile robot operations
  - Network security measures for robot communications
  - Latency considerations for real-time control
- **Router Setup**:
  - Creating isolated network for robot systems
  - Quality of Service (QoS) settings for priority traffic

## Diagrams

### Network Architecture Block Diagram
Create a block diagram showing the interconnection of components:
```
Simulation PC (Ubuntu 22.04 + ROS 2 Humble + RTX GPU)
           |
           | (Ethernet/WiFi)
           |
         Router (With QoS)
           |
           | (Ethernet for reliability)
           |
Jetson Orin (Robot Brain running ROS 2 nodes)
           |
           | (USB/Serial)
           |
Robot Arm (with Dynamixel servos and sensors)
```

The diagram should illustrate:
- Simulation PC as the primary development workstation
- Router managing network traffic between all components
- Jetson Orin as the robot's "brain" processing commands and sensor data
- Robot arm as the physical component executing tasks
- Bidirectional arrows showing data flow between components
- Different connection types (Ethernet for reliability, WiFi for mobility)

## Required Software Installation Steps
- Install ROS 2 Humble Hawksbill on all machines
- Set up NVIDIA drivers and CUDA toolkit on simulation PC
- Configure network settings for multi-machine ROS setup
- Install camera and robot arm drivers on Jetson Orin

## Assessment Criteria
Students will be evaluated on their ability to:
- Successfully configure and connect all hardware components
- Demonstrate proper network setup between devices
- Validate hardware functionality with simple test programs
- Explain the rationale behind each hardware selection