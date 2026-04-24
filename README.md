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
Trong đó:

x_n: tọa độ UAV theo trục x;
y_n: tọa độ UAV theo trục y;
h_n: độ cao của UAV.

Các thiết bị IoT và jammer được giả sử đặt cố định trên mặt đất, trong khi UAV có thể di chuyển trong không gian ba chiều.

English

The considered system consists of the following main components:

UAV: an unmanned aerial vehicle acting as a mobile data collector.
IoT devices: ground sensor nodes with fixed positions.
Jammers: wireless interference sources that affect communication quality.
Obstacles: buildings or urban structures that may block the communication path or create collision risks.

The UAV position at time step n is represented in 3D space as:

q_n = [x_n, y_n, h_n]

where:

x_n: UAV coordinate on the x-axis;
y_n: UAV coordinate on the y-axis;
h_n: UAV altitude.

IoT devices and jammers are assumed to be fixed on the ground, while the UAV moves in the 3D space.

6. Không gian hành động / Action Space
Tiếng Việt

Tại mỗi bước thời gian, UAV chọn một hành động di chuyển trong không gian ba chiều. Hành động của UAV được biểu diễn như sau:

a_n = [a_x, a_y, a_z]

Trong đó:

a_x, a_y, a_z ∈ {-1, 0, 1}

Ý nghĩa:

-1: UAV di chuyển một bước theo chiều âm của trục tương ứng.
0: UAV giữ nguyên theo trục tương ứng.
1: UAV di chuyển một bước theo chiều dương của trục tương ứng.

Sau khi chọn hành động, vị trí UAV được cập nhật theo công thức:

q_{n+1} = q_n + a_n
English

At each time step, the UAV selects a movement action in the 3D space. The UAV action is represented as:

a_n = [a_x, a_y, a_z]

where:

a_x, a_y, a_z ∈ {-1, 0, 1}

Meaning:

-1: the UAV moves one step in the negative direction of the corresponding axis.
0: the UAV remains unchanged along the corresponding axis.
1: the UAV moves one step in the positive direction of the corresponding axis.

After selecting an action, the UAV position is updated as:

q_{n+1} = q_n + a_n
7. Mô hình kênh truyền / Communication Channel Model
Tiếng Việt

Chất lượng kênh truyền giữa UAV và thiết bị IoT phụ thuộc vào nhiều yếu tố như khoảng cách, trạng thái truyền thẳng LoS, trạng thái không truyền thẳng NLoS, vật cản và nhiễu từ jammer.

Trong hệ thống, chỉ số SINR được sử dụng để đánh giá chất lượng liên kết truyền thông. SINR càng cao thì chất lượng liên kết càng tốt và UAV có thể thu thập được nhiều dữ liệu hơn.

Thông lượng truyền dữ liệu được xác định dựa trên công thức Shannon:

R = B log2(1 + SINR)

Trong đó:

R: thông lượng truyền dữ liệu;
B: băng thông kênh truyền;
SINR: tỷ số tín hiệu trên nhiễu cộng tạp âm.
English

The communication channel quality between the UAV and an IoT device depends on several factors, including distance, Line-of-Sight (LoS), Non-Line-of-Sight (NLoS), obstacles, and jammer interference.

In this system, SINR is used to evaluate the communication link quality. A higher SINR indicates a better communication link and allows the UAV to collect more data.

The data throughput is calculated based on Shannon's formula:

R = B log2(1 + SINR)

where:

R: data throughput;
B: channel bandwidth;
SINR: signal-to-interference-plus-noise ratio.
8. Phát biểu bài toán / Problem Formulation
Tiếng Việt

Mục tiêu của bài toán là tìm quỹ đạo bay tối ưu cho UAV nhằm cực đại hóa tổng lượng dữ liệu thu thập được từ các thiết bị IoT trong suốt quá trình bay.

Bài toán cần thỏa mãn các ràng buộc chính:

UAV không được va chạm với vật cản.
UAV không được bay ra ngoài vùng hoạt động.
UAV phải còn đủ năng lượng để tiếp tục nhiệm vụ.
Chất lượng liên kết truyền thông phải đủ tốt.
Tại mỗi thời điểm, UAV chỉ thu thập dữ liệu từ tối đa một thiết bị IoT.

Như vậy, bài toán không chỉ là bài toán di chuyển hình học, mà là bài toán tối ưu tổng hợp giữa truyền thông vô tuyến, năng lượng và điều khiển thông minh.

English

The objective of the problem is to find an optimal UAV trajectory that maximizes the total amount of data collected from IoT devices during the mission.

The problem must satisfy several main constraints:

The UAV must avoid collisions with obstacles.
The UAV must remain inside the allowed operation area.
The UAV must have enough energy to continue the mission.
The communication link quality must be sufficient.
At each time step, the UAV can collect data from at most one IoT device.

Therefore, this is not only a geometric path planning problem, but also a joint optimization problem involving wireless communication, energy consumption, and intelligent control.

9. Mô hình MDP / Markov Decision Process
Tiếng Việt

Bài toán tối ưu quỹ đạo UAV được mô hình hóa dưới dạng Markov Decision Process (MDP). MDP giúp chuyển bài toán điều khiển quỹ đạo phức tạp thành một bài toán ra quyết định tuần tự theo thời gian.

MDP được biểu diễn bởi:

MDP = <S, A, R, P>

Trong đó:

S: không gian trạng thái;
A: không gian hành động;
R: hàm phần thưởng;
P: xác suất chuyển trạng thái.

Trạng thái của UAV bao gồm:

vị trí hiện tại của UAV;
năng lượng còn lại;
lượng dữ liệu đã thu thập;
thông tin về thiết bị IoT;
khoảng cách giữa UAV và các thiết bị IoT;
chất lượng liên kết truyền thông;
thông tin vật cản và vùng quan sát môi trường.
English

The UAV trajectory optimization problem is modeled as a Markov Decision Process (MDP). MDP converts the complex trajectory control problem into a sequential decision-making problem over time.

The MDP is represented as:

MDP = <S, A, R, P>

where:

S: state space;
A: action space;
R: reward function;
P: state transition probability.

The UAV state includes:

current UAV position;
remaining energy;
collected data;
IoT device information;
distance between the UAV and IoT devices;
communication link quality;
obstacle information and environmental observation.
10. Hàm thưởng / Reward Function
Tiếng Việt

Hàm thưởng được thiết kế để hướng UAV học được quỹ đạo bay tốt. Reward tổng tại mỗi bước thời gian gồm ba phần chính:

r_n = r_data - r_energy - r_safety

Trong đó:

r_data: phần thưởng khi UAV thu thập được dữ liệu;
r_energy: phần phạt do tiêu hao năng lượng khi di chuyển;
r_safety: phần phạt khi UAV đi vào vùng nguy hiểm, va chạm vật cản hoặc bay ra ngoài phạm vi cho phép.

Cách thiết kế này giúp UAV không chỉ tập trung vào việc thu thập thật nhiều dữ liệu, mà còn phải học cách bay an toàn và tiết kiệm năng lượng.

English

The reward function is designed to guide the UAV toward learning a good flight trajectory. The total reward at each time step consists of three main parts:

r_n = r_data - r_energy - r_safety

where:

r_data: reward for collecting data;
r_energy: penalty for energy consumption during movement;
r_safety: penalty for entering unsafe areas, colliding with obstacles, or flying outside the allowed region.

This reward design encourages the UAV not only to collect more data, but also to fly safely and energy-efficiently.

11. Thuật toán DDQN / DDQN Algorithm
Tiếng Việt

DDQN là phiên bản cải tiến của DQN, được sử dụng để giảm hiện tượng đánh giá quá mức giá trị Q. Điểm khác biệt quan trọng của DDQN là tách riêng hai bước:

Chọn hành động bằng mạng online.
Đánh giá hành động bằng mạng target.

Giá trị mục tiêu của DDQN có dạng:

y = r + γ Q_target(s', argmax_a Q_online(s', a))

Trong đó:

r: phần thưởng nhận được;
γ: hệ số chiết khấu;
s': trạng thái kế tiếp;
Q_online: mạng online dùng để chọn hành động;
Q_target: mạng target dùng để đánh giá hành động.

Nhờ cơ chế này, DDQN giúp quá trình huấn luyện ổn định hơn và hỗ trợ UAV học quỹ đạo bay hiệu quả hơn trong môi trường phức tạp.

English

DDQN is an improved version of DQN, designed to reduce the overestimation problem of Q-values. The key idea of DDQN is to separate two steps:

Action selection using the online network.
Action evaluation using the target network.

The DDQN target value is defined as:

y = r + γ Q_target(s', argmax_a Q_online(s', a))

where:

r: received reward;
γ: discount factor;
s': next state;
Q_online: online network used for action selection;
Q_target: target network used for action evaluation.

With this mechanism, DDQN improves training stability and helps the UAV learn a more effective trajectory in complex environments.

12. Quy trình huấn luyện / Training Process
Tiếng Việt

Quy trình huấn luyện DDQN gồm các bước chính:

Khởi tạo môi trường mô phỏng.
Sinh ngẫu nhiên vị trí UAV, thiết bị IoT, jammer và vật cản.
Khởi tạo mạng online, mạng target và bộ nhớ kinh nghiệm.
UAV quan sát trạng thái hiện tại của môi trường.
UAV chọn hành động theo chính sách epsilon-greedy.
Môi trường cập nhật vị trí UAV và trạng thái mới.
Tính reward dựa trên dữ liệu thu được, năng lượng và độ an toàn.
Lưu mẫu kinh nghiệm vào replay memory.
Lấy minibatch từ bộ nhớ để huấn luyện mạng online.
Cập nhật mạng target theo chu kỳ.
Lặp lại quá trình cho đến khi mô hình học được chính sách bay tốt.
English

The DDQN training process includes the following main steps:

Initialize the simulation environment.
Randomly generate UAV, IoT device, jammer, and obstacle positions.
Initialize the online network, target network, and replay memory.
The UAV observes the current environment state.
The UAV selects an action using the epsilon-greedy policy.
The environment updates the UAV position and next state.
The reward is calculated based on collected data, energy consumption, and safety.
The experience sample is stored in replay memory.
A minibatch is sampled from memory to train the online network.
The target network is updated periodically.
The process is repeated until the model learns a good flight policy.

