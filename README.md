# UAV-IoT DDQN 3D Trajectory Optimization

## Tên đề tài / Project Title

**Nghiên cứu và triển khai hệ thống UAV hỗ trợ thu thập dữ liệu trong mạng IoT sử dụng thuật toán Double Deep Q-Network để tối ưu quỹ đạo bay 3D**

**English title:**  
**Research and Implementation of a UAV-Assisted Data Collection System in IoT Networks using Double Deep Q-Network for 3D Trajectory Optimization**

---

## 1. Giới thiệu tiếng Việt

Đề tài nghiên cứu và triển khai một hệ thống UAV hỗ trợ thu thập dữ liệu trong mạng Internet vạn vật (IoT). Trong hệ thống này, UAV đóng vai trò như một nút thu thập dữ liệu di động, bay trong môi trường ba chiều để thu thập dữ liệu từ các thiết bị IoT được đặt cố định trên mặt đất.

Trong môi trường thực tế, UAV không chỉ cần bay đến gần các thiết bị IoT để tăng lượng dữ liệu thu thập, mà còn phải đối mặt với nhiều yếu tố phức tạp như vật cản, nguồn gây nhiễu, giới hạn năng lượng và sự suy giảm chất lượng kênh truyền. Vì vậy, bài toán tối ưu quỹ đạo UAV không đơn thuần là tìm đường bay ngắn nhất, mà là tìm một chiến lược bay hợp lý để cân bằng giữa thu thập dữ liệu, tránh va chạm, giảm nhiễu và tiết kiệm năng lượng.

Để giải quyết bài toán này, đề tài sử dụng thuật toán **Double Deep Q-Network (DDQN)**, một phương pháp học tăng cường sâu có khả năng học chính sách điều khiển thông qua quá trình tương tác giữa UAV và môi trường.

---

## 2. English Introduction

This project studies and implements a UAV-assisted data collection system for Internet of Things (IoT) networks. In this system, the UAV acts as a mobile data collector that flies in a three-dimensional environment to collect data from ground IoT devices.

In practical environments, the UAV must not only move close to IoT devices to improve data collection performance, but also deal with obstacles, wireless jammers, limited energy, and varying communication channel quality. Therefore, UAV trajectory optimization is not simply a shortest-path problem. Instead, it is a multi-objective decision-making problem that must balance data collection, obstacle avoidance, interference reduction, and energy efficiency.

To solve this problem, this project applies the **Double Deep Q-Network (DDQN)** algorithm, a deep reinforcement learning method that allows the UAV to learn an effective flight policy through interaction with the environment.

---

## 3. Mục tiêu đề tài / Project Objectives

### Tiếng Việt

Mục tiêu chính của đề tài là xây dựng và mô phỏng một hệ thống UAV hỗ trợ thu thập dữ liệu trong mạng IoT, trong đó UAV sử dụng thuật toán DDQN để học quỹ đạo bay tối ưu trong không gian 3D.

Các mục tiêu cụ thể gồm:

- Xây dựng mô hình hệ thống UAV-IoT trong môi trường ba chiều.
- Mô hình hóa các thiết bị IoT, nguồn gây nhiễu và vật cản.
- Xây dựng mô hình kênh truyền giữa UAV và thiết bị IoT.
- Phát biểu bài toán cực đại hóa lượng dữ liệu thu thập.
- Chuyển bài toán tối ưu quỹ đạo UAV thành mô hình Markov Decision Process (MDP).
- Áp dụng thuật toán DDQN để huấn luyện UAV lựa chọn hành động bay phù hợp.
- Mô phỏng, trực quan hóa và đánh giá kết quả quỹ đạo bay của UAV.

### English

The main objective of this project is to build and simulate a UAV-assisted IoT data collection system, where the UAV uses the DDQN algorithm to learn an optimized 3D flight trajectory.

The specific objectives are:

- Build a UAV-IoT system model in a 3D environment.
- Model IoT devices, wireless jammers, and obstacles.
- Develop the wireless channel model between the UAV and IoT devices.
- Formulate the data throughput maximization problem.
- Convert the UAV trajectory optimization problem into a Markov Decision Process (MDP).
- Apply the DDQN algorithm to train the UAV to select suitable movement actions.
- Simulate, visualize, and evaluate the UAV trajectory results.

---

## 4. Bối cảnh nghiên cứu / Research Background

### Tiếng Việt

Trong các hệ thống IoT hiện đại, số lượng thiết bị cảm biến ngày càng tăng nhanh. Các thiết bị này thường được triển khai trong những môi trường rộng, phức tạp hoặc khó xây dựng hạ tầng truyền thông cố định. Do giới hạn về năng lượng, công suất phát và khoảng cách truyền, việc truyền dữ liệu trực tiếp từ thiết bị IoT về trạm gốc có thể gặp nhiều khó khăn.

UAV là một giải pháp phù hợp vì có khả năng di chuyển linh hoạt và có thể bay đến gần các thiết bị IoT để thu thập dữ liệu. Nhờ đó, UAV có thể cải thiện chất lượng truyền thông, mở rộng vùng phục vụ và giảm phụ thuộc vào hạ tầng mạng cố định.

Tuy nhiên, UAV cũng có nhiều giới hạn như dung lượng pin, thời gian bay, độ cao hoạt động, nguy cơ va chạm với vật cản và ảnh hưởng của nhiễu vô tuyến. Vì vậy, cần có một phương pháp điều khiển thông minh để UAV có thể tự học quỹ đạo bay phù hợp trong môi trường phức tạp.

### English

In modern IoT systems, the number of sensor devices is increasing rapidly. These devices are often deployed in large, complex, or infrastructure-limited environments. Due to limited energy, transmission power, and communication distance, direct data transmission from IoT devices to a fixed base station may be difficult.

UAVs are a promising solution because they can move flexibly and fly close to IoT devices to collect data. As a result, UAVs can improve communication quality, extend service coverage, and reduce dependence on fixed network infrastructure.

However, UAVs also face several limitations, such as battery capacity, flight time, altitude constraints, collision risks, and wireless interference. Therefore, an intelligent control method is required so that the UAV can learn an effective trajectory in complex environments.

---

## 5. Mô hình hệ thống / System Model

### Tiếng Việt

Hệ thống được xét trong đề tài gồm các thành phần chính:

- **UAV**: thiết bị bay không người lái, đóng vai trò là nút thu thập dữ liệu di động.
- **Thiết bị IoT**: các nút cảm biến được đặt cố định trên mặt đất.
- **Jammer**: các nguồn gây nhiễu vô tuyến, ảnh hưởng đến chất lượng liên kết truyền thông.
- **Vật cản**: các tòa nhà hoặc công trình trong môi trường đô thị, có thể gây che khuất đường truyền hoặc gây nguy cơ va chạm.

Vị trí của UAV tại thời điểm `n` được biểu diễn trong không gian ba chiều:

```text
q_n = [x_n, y_n, h_n]
