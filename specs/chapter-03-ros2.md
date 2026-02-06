# Specification: Chapter 3 - ROS 2 Basics

## Learning Objectives
- Understand the "Graph Metaphor": Nodes, Topics, and Messages.
- Master the Publisher-Subscriber communication model.
- Set up a ROS 2 Workspace (`colcon_ws`).
- Write your first Python Node.

## Core Concepts
1. **The Graph**: ROS 2 is like a network of brain cells (Nodes) talking via nerves (Topics).
2. **Nodes**: Single-purpose executable programs (e.g., "CameraDriver", "MotionPlanner").
3. **Topics**: The pipes through which data flows. One node speaks (Publishes), another listens (Subscribes).

## Hands-On Implementation
1. **Setup Workspace**:
   - `mkdir -p ~/ros2_ws/src`
   - `colcon build`
2. **Coding**:
   - Create `py_pubsub` package.
   - Write `publisher_member_function.py` (Talker).
   - Write `subscriber_member_function.py` (Listener).

## Diagrams
- **Pub/Sub Diagram**: Visualizing a "Talker" node sending "Hello World" string to a "Listener" node over the "/topic".

## Essential Commands
- `ros2 node list`
- `ros2 topic list`
- `ros2 run <package> <executable>`