# UAV-IoT DDQN 3D Trajectory Optimization

## 1. Project Title

**Nghiên cứu và triển khai hệ thống UAV hỗ trợ thu thập dữ liệu trong mạng IoT sử dụng thuật toán Double Deep Q-Network để tối ưu quỹ đạo bay 3D**

**English title:**  
**UAV-Assisted Data Collection in IoT Networks using Double Deep Q-Network for 3D Trajectory Optimization**

---

## 2. Overview

This project studies and implements a UAV-assisted data collection system for Internet of Things (IoT) networks. The UAV acts as a mobile data collector that flies in a three-dimensional environment to collect data from ground IoT devices.

The main objective is to optimize the UAV's 3D trajectory so that it can collect as much data as possible while avoiding obstacles, reducing the effect of wireless interference, and saving energy. The trajectory optimization problem is modeled as a Markov Decision Process (MDP), and a Double Deep Q-Network (DDQN) algorithm is applied to learn an efficient flight policy.

---

## 3. Motivation

In modern IoT networks, many sensors are deployed in large or complex environments such as smart cities, industrial areas, campuses, disaster zones, and remote monitoring regions. Direct data transmission from IoT devices to a fixed base station may be difficult because of limited transmission power, weak communication links, obstacles, and interference.

UAVs provide a flexible solution because they can move close to IoT devices and collect data directly. However, UAV trajectory planning is challenging because the UAV must simultaneously consider:

- 3D movement space;
- limited battery energy;
- obstacle avoidance;
- wireless channel quality;
- interference from jammers;
- data collection efficiency;
- unknown or changing environments.

Therefore, this project uses deep reinforcement learning, especially DDQN, to allow the UAV to learn an intelligent trajectory planning strategy.

---

## 4. Main Contributions

The main contributions of this project are:

1. A 3D UAV-IoT data collection scenario is constructed with ground IoT devices, jammers, and obstacles.

2. The UAV trajectory optimization problem is formulated as a Markov Decision Process including state space, action space, transition process, and reward function.

3. A DDQN-based trajectory optimization algorithm is implemented to learn a flight policy for the UAV.

4. The simulation evaluates the UAV trajectory in a 3D environment and visualizes the learned movement path.

5. The project provides a foundation for future deployment in smart cities, industrial IoT, emergency communication, and 6G-oriented UAV communication systems.

---

## 5. System Model

The system includes one UAV, multiple ground IoT devices, several wireless jammers, and obstacles in a 3D environment.

The UAV is modeled as a mobile data collector. At each time step, the UAV observes the current environment state, selects an action, updates its position, and receives a reward.

The UAV position is represented as:

```text
q_n = [x_n, y_n, h_n]
