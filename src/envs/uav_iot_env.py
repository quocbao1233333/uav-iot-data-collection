from typing import Dict
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import ScenarioConfig, build_action_table


class UAVIoTEnv(gym.Env):
    """
    UAV-assisted IoT data collection environment (3D+5O).

    This environment follows:
    - Chapter 2: system model + channel model
    - Chapter 3: MDP state / reward logic
    - Chapter 4: fixed simulation scenario

    Key fixes versus the previous version:
    1. UAV start position is randomized in x, y, z.
    2. No-op action (0,0,0) is removed.
    3. Reward shaping encourages movement toward useful communication states.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: ScenarioConfig, seed: int = 42):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.action_table = np.array(
            build_action_table(cfg.step_xy, cfg.step_z), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_table))

        # state = s1 + s2 + s3
        # s1: q_n (3) + b_n (1) + L_n (1)
        # s2 per device: q_u (3) + l_u (1) + sinr_u (1) + d_u (1) + D_u (1)
        # s3: current obs (6) + lookahead obs (6)
        self.state_dim = 3 + 1 + 1 + cfg.num_devices * 7 + 12
        high = np.full((self.state_dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._reset_internal_state()

    def _reset_internal_state(self):
        self.step_count = 0
        self.done = False
        self.last_action_vec = np.zeros(3, dtype=np.float32)

        self.uav_pos = np.zeros(3, dtype=np.float32)
        self.battery = self.cfg.battery_init
        self.collected_bits = 0.0

        self.device_positions = np.zeros((self.cfg.num_devices, 3), dtype=np.float32)
        self.jammer_positions = np.zeros((self.cfg.num_jammers, 3), dtype=np.float32)
        self.jammer_powers = np.zeros(self.cfg.num_jammers, dtype=np.float32)
        self.device_rate_caps = np.zeros(self.cfg.num_devices, dtype=np.float32)
        self.device_remaining_bits = np.full(
            self.cfg.num_devices,
            self.cfg.initial_data_bits_per_device,
            dtype=np.float32,
        )

        self.last_distances = np.zeros(self.cfg.num_devices, dtype=np.float32)
        self.last_sinrs = np.zeros(self.cfg.num_devices, dtype=np.float32)
        self.last_links = np.zeros(self.cfg.num_devices, dtype=np.float32)

        self.prev_best_distance = None
        self.prev_best_sinr = None

        self.trajectory = []
        self.link_history = []
        self.reward_history = []
        self.action_history = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_internal_state()
        self._spawn_scene()

        self.last_distances, self.last_sinrs, self.last_links, collected_now = \
            self._communication_step(self.uav_pos)
        self.collected_bits += collected_now

        self.prev_best_distance = self._best_remaining_device_distance()
        self.prev_best_sinr = self._best_remaining_device_sinr()

        state = self._build_state()
        info = self._build_info(collision=False, out_of_bounds=False)
        return state, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        action_vec = self.action_table[action]
        proposed_pos = self.uav_pos + action_vec

        collision = self._intersects_obstacle(proposed_pos)
        out_of_bounds = not self._inside_bounds(proposed_pos)

        if collision or out_of_bounds:
            new_pos = self.uav_pos.copy()
        else:
            new_pos = proposed_pos

        movement_distance = float(np.linalg.norm(new_pos - self.uav_pos))
        self.uav_pos = new_pos
        self.last_action_vec = action_vec
        self.step_count += 1
        self.action_history.append(action)

        # Communication update after movement
        self.last_distances, self.last_sinrs, self.last_links, collected_now = \
            self._communication_step(self.uav_pos)
        self.collected_bits += collected_now

        current_best_distance = self._best_remaining_device_distance()
        current_best_sinr = self._best_remaining_device_sinr()

        distance_improvement = 0.0
        sinr_improvement = 0.0
        if self.prev_best_distance is not None and current_best_distance is not None:
            distance_improvement = self.prev_best_distance - current_best_distance

        if self.prev_best_sinr is not None and current_best_sinr is not None:
            sinr_improvement = np.log1p(current_best_sinr) - np.log1p(self.prev_best_sinr)

        energy_cost = self.cfg.move_energy_base + \
                      self.cfg.move_energy_per_meter * movement_distance
        self.battery -= energy_cost

        reward = self._compute_reward(
            collected_bits_step=collected_now,
            energy_cost=energy_cost,
            collision=collision,
            out_of_bounds=out_of_bounds,
            link_ok=np.any(self.last_links > 0.0),
            distance_improvement=distance_improvement,
            sinr_improvement=sinr_improvement,
        )

        self.reward_history.append(reward)
        self.trajectory.append(self.uav_pos.copy())
        self.link_history.append(self.last_links.copy())

        self.prev_best_distance = current_best_distance
        self.prev_best_sinr = current_best_sinr

        if self.battery <= 0:
            self.done = True
        if np.all(self.device_remaining_bits <= 1e-6):
            self.done = True
        if self.step_count >= self.cfg.max_steps:
            self.done = True

        state = self._build_state()
        info = self._build_info(collision=collision, out_of_bounds=out_of_bounds)
        return state, reward, self.done, False, info

    # --------------------------
    # Scene generation
    # --------------------------
    def _spawn_scene(self):
        x_min, x_max, y_min, y_max = self.cfg.uav_start_xy_range
        z0 = float(self.rng.uniform(*self.cfg.uav_start_z_range))

        while True:
            x0 = float(self.rng.uniform(x_min, x_max))
            y0 = float(self.rng.uniform(y_min, y_max))
            candidate = np.array([x0, y0, z0], dtype=np.float32)
            if not self._intersects_obstacle(candidate):
                self.uav_pos = candidate
                break

        self.device_positions = self._sample_ground_points(self.cfg.num_devices)
        self.jammer_positions = self._sample_ground_points(self.cfg.num_jammers)

        self.device_rate_caps = self.rng.uniform(
            self.cfg.per_device_rate_range_bps[0],
            self.cfg.per_device_rate_range_bps[1],
            size=self.cfg.num_devices
        ).astype(np.float32)

        self.device_remaining_bits[:] = self.cfg.initial_data_bits_per_device

        self.jammer_powers = self.rng.uniform(
            self.cfg.jammer_power_range_w[0],
            self.cfg.jammer_power_range_w[1],
            size=self.cfg.num_jammers
        ).astype(np.float32)

        self.trajectory = [self.uav_pos.copy()]
        self.link_history = []
        self.action_history = []

    def _sample_ground_points(self, n: int) -> np.ndarray:
        pts = []
        while len(pts) < n:
            x = float(self.rng.uniform(5.0, self.cfg.x_max - 5.0))
            y = float(self.rng.uniform(5.0, self.cfg.y_max - 5.0))
            p = np.array([x, y, 0.0], dtype=np.float32)
            if not self._inside_any_obstacle_footprint(p[:2]):
                pts.append(p)
        return np.stack(pts, axis=0)

    # --------------------------
    # Communication model
    # --------------------------
    def _communication_step(self, uav_pos: np.ndarray):
        distances = np.zeros(self.cfg.num_devices, dtype=np.float32)
        sinrs = np.zeros(self.cfg.num_devices, dtype=np.float32)
        links = np.zeros(self.cfg.num_devices, dtype=np.float32)

        for u in range(self.cfg.num_devices):
            d_u = np.linalg.norm(uav_pos - self.device_positions[u])
            distances[u] = float(d_u)

            g_u_db = self._channel_gain_db(uav_pos, self.device_positions[u])
            signal_power = self.cfg.device_tx_power_w * (10.0 ** (g_u_db / 10.0))

            interference = 0.0
            for j in range(self.cfg.num_jammers):
                g_j_db = self._channel_gain_db(uav_pos, self.jammer_positions[j])
                interference += self.jammer_powers[j] * (10.0 ** (g_j_db / 10.0))

            noise_w = self._noise_power_w()
            sinrs[u] = float(signal_power / (interference + noise_w + 1e-12))

        valid = np.where(self.device_remaining_bits > 1e-6)[0]
        collected_bits_step = 0.0

        if len(valid) > 0:
            best_idx = valid[np.argmax(sinrs[valid])]
            if sinrs[best_idx] >= self.cfg.sinr_threshold_linear:
                links[best_idx] = 1.0
                throughput_bps = self.cfg.bandwidth_hz * np.log2(1.0 + sinrs[best_idx])
                effective_bps = min(float(throughput_bps), float(self.device_rate_caps[best_idx]))
                bits_now = effective_bps * self.cfg.delta_t
                bits_now = min(bits_now, float(self.device_remaining_bits[best_idx]))
                self.device_remaining_bits[best_idx] -= bits_now
                collected_bits_step = bits_now

        return distances, sinrs, links, float(collected_bits_step)

    def _channel_gain_db(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        d = float(np.linalg.norm(uav_pos - ground_pos))
        los = self._is_los(uav_pos, ground_pos)

        if los:
            return self.cfg.beta_los_db + \
                   self.cfg.alpha_los * np.log10(max(d, 1.0)) + \
                   self.cfg.eta_los

        return self.cfg.beta_nlos_db + \
               self.cfg.alpha_nlos * np.log10(max(d, 1.0)) + \
               self.cfg.eta_nlos

    def _is_los(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> bool:
        samples = 40
        for t in np.linspace(0.0, 1.0, samples):
            point = ground_pos * (1.0 - t) + uav_pos * t
            x, y, z = point
            for obs in self.cfg.obstacles:
                if obs.x <= x <= obs.x + obs.width and obs.y <= y <= obs.y + obs.length:
                    if obs.height >= z:
                        return False
        return True

    def _noise_power_w(self) -> float:
        total_dbm = self.cfg.noise_psd_dbm_per_hz + 10.0 * np.log10(self.cfg.bandwidth_hz)
        return 1e-3 * (10.0 ** (total_dbm / 10.0))

    # --------------------------
    # Reward
    # --------------------------
    def _compute_reward(
        self,
        collected_bits_step: float,
        energy_cost: float,
        collision: bool,
        out_of_bounds: bool,
        link_ok: bool,
        distance_improvement: float,
        sinr_improvement: float,
    ) -> float:
        # r_n = r_n1 - r_n2 - r_n3 + shaping
        r_data = self.cfg.reward_data_scale * collected_bits_step
        r_link = self.cfg.reward_link_bonus if link_ok else 0.0
        r_dist = self.cfg.reward_distance_gain * max(0.0, distance_improvement)
        r_sinr = self.cfg.reward_sinr_gain * max(0.0, sinr_improvement)

        p_energy = self.cfg.penalty_energy_scale * energy_cost
        p_safety = 0.0
        if collision:
            p_safety += self.cfg.penalty_collision
        if out_of_bounds:
            p_safety += self.cfg.penalty_out_of_bounds
        if not link_ok:
            p_safety += self.cfg.penalty_no_link
        if distance_improvement <= 0 and not link_ok:
            p_safety += self.cfg.penalty_no_progress

        return float(r_data + r_link + r_dist + r_sinr - p_energy - p_safety)

    # --------------------------
    # State
    # --------------------------
    def _build_state(self) -> np.ndarray:
        # s_{n,1} = {q_n, b_n, L_n}
        s1 = np.array([
            self.uav_pos[0] / self.cfg.x_max,
            self.uav_pos[1] / self.cfg.y_max,
            self.uav_pos[2] / self.cfg.z_max,
            self.battery / self.cfg.battery_init,
            self.collected_bits / (self.cfg.num_devices * self.cfg.initial_data_bits_per_device),
        ], dtype=np.float32)

        # s_{n,2} = {q_u, l_u, SINR_u, d_u, D_u}
        pieces = []
        norm_dist = np.linalg.norm([self.cfg.x_max, self.cfg.y_max, self.cfg.z_max])

        for u in range(self.cfg.num_devices):
            pos = self.device_positions[u]
            pieces.extend([
                pos[0] / self.cfg.x_max,
                pos[1] / self.cfg.y_max,
                0.0,
                self.last_links[u],
                np.log1p(self.last_sinrs[u]) / np.log(1.0 + 1000.0),
                self.last_distances[u] / norm_dist,
                self.device_remaining_bits[u] / self.cfg.initial_data_bits_per_device,
            ])
        s2 = np.array(pieces, dtype=np.float32)

        # s_{n,3} = {o_n, o_{n+1}}
        o_n = self._observation_features(self.uav_pos)
        lookahead_pos = self.uav_pos + self.last_action_vec
        lookahead_pos = np.clip(
            lookahead_pos,
            [0.0, 0.0, 0.0],
            [self.cfg.x_max, self.cfg.y_max, self.cfg.z_max],
        )
        o_n1 = self._observation_features(lookahead_pos)
        s3 = np.concatenate([o_n, o_n1]).astype(np.float32)

        return np.concatenate([s1, s2, s3]).astype(np.float32)

    def _observation_features(self, pos: np.ndarray) -> np.ndarray:
        x, y, z = pos

        boundary_distances = [
            self.cfg.x_max - x,
            x,
            self.cfg.y_max - y,
            y,
            self.cfg.z_max - z,
            z,
        ]

        obstacle_margins = []
        for direction in range(6):
            margin = self.cfg.obs_clip_distance
            for obs in self.cfg.obstacles:
                if direction == 0 and obs.y <= y <= obs.y + obs.length and z <= obs.height and x <= obs.x:
                    margin = min(margin, obs.x - x)
                elif direction == 1 and obs.y <= y <= obs.y + obs.length and z <= obs.height and x >= obs.x + obs.width:
                    margin = min(margin, x - (obs.x + obs.width))
                elif direction == 2 and obs.x <= x <= obs.x + obs.width and z <= obs.height and y <= obs.y:
                    margin = min(margin, obs.y - y)
                elif direction == 3 and obs.x <= x <= obs.x + obs.width and z <= obs.height and y >= obs.y + obs.length:
                    margin = min(margin, y - (obs.y + obs.length))
                elif direction == 4:
                    if obs.x <= x <= obs.x + obs.width and obs.y <= y <= obs.y + obs.length and obs.height >= z:
                        margin = min(margin, obs.height - z)
                elif direction == 5:
                    margin = min(margin, z)
            obstacle_margins.append(min(boundary_distances[direction], margin))

        features = np.array(obstacle_margins, dtype=np.float32) / self.cfg.obs_clip_distance
        return np.clip(features, 0.0, 1.0)

    # --------------------------
    # Helper metrics
    # --------------------------
    def _best_remaining_device_distance(self):
        valid = np.where(self.device_remaining_bits > 1e-6)[0]
        if len(valid) == 0:
            return None
        return float(np.min(self.last_distances[valid]))

    def _best_remaining_device_sinr(self):
        valid = np.where(self.device_remaining_bits > 1e-6)[0]
        if len(valid) == 0:
            return None
        return float(np.max(self.last_sinrs[valid]))

    # --------------------------
    # Geometry
    # --------------------------
    def _inside_bounds(self, pos: np.ndarray) -> bool:
        return (
            0.0 <= pos[0] <= self.cfg.x_max and
            0.0 <= pos[1] <= self.cfg.y_max and
            0.0 <= pos[2] <= self.cfg.z_max
        )

    def _inside_any_obstacle_footprint(self, xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        for obs in self.cfg.obstacles:
            if obs.x <= x <= obs.x + obs.width and obs.y <= y <= obs.y + obs.length:
                return True
        return False

    def _intersects_obstacle(self, pos: np.ndarray) -> bool:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        for obs in self.cfg.obstacles:
            if obs.x <= x <= obs.x + obs.width and obs.y <= y <= obs.y + obs.length and z <= obs.height:
                return True
        return False

    def _build_info(self, collision: bool, out_of_bounds: bool) -> Dict:
        return {
            "uav_pos": self.uav_pos.copy(),
            "battery": float(self.battery),
            "collected_bits": float(self.collected_bits),
            "device_remaining_bits": self.device_remaining_bits.copy(),
            "device_positions": self.device_positions.copy(),
            "jammer_positions": self.jammer_positions.copy(),
            "jammer_powers": self.jammer_powers.copy(),
            "links": self.last_links.copy(),
            "sinrs": self.last_sinrs.copy(),
            "distances": self.last_distances.copy(),
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "trajectory": np.stack(self.trajectory, axis=0) if len(self.trajectory) > 0 else np.empty((0, 3)),
            "link_history": np.stack(self.link_history, axis=0) if len(self.link_history) > 0 else np.empty((0, self.cfg.num_devices)),
            "action_history": np.array(self.action_history, dtype=np.int32),
        }
        
        
        
