from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import itertools


@dataclass
class Obstacle3D:
    x: float
    y: float
    width: float
    length: float
    height: float


@dataclass
class ScenarioConfig:
    # Chapter 4 fixed scene size
    x_max: float = 50.0
    y_max: float = 50.0
    z_max: float = 10.0

    # Fixed 3D+5O
    num_devices: int = 3
    num_jammers: int = 3
    num_obstacles: int = 5

    # Mission
    delta_t: float = 1.0
    max_steps: int = 120

    # UAV random start range (fixed previous version's issue)
    uav_start_xy_range: Tuple[float, float, float, float] = (5.0, 45.0, 5.0, 45.0)
    uav_start_z_range: Tuple[float, float] = (3.0, 10.0)

    # Action step size
    step_xy: float = 1.0
    step_z: float = 1.0

    # Battery model
    battery_init: float = 220.0
    move_energy_base: float = 0.15
    move_energy_per_meter: float = 0.12

    # Chapter 4 communication parameters
    bandwidth_hz: float = 100e6
    device_tx_power_w: float = 43.0
    jammer_power_range_w: Tuple[float, float] = (10.0, 50.0)
    noise_psd_dbm_per_hz: float = -60.0

    beta_los_db: float = -30.0
    beta_nlos_db: float = -35.0
    eta_los: float = 1.41
    eta_nlos: float = 2.23
    alpha_los: float = -2.5
    alpha_nlos: float = -3.04

    # Link threshold
    sinr_threshold_linear: float = 0.5  # relaxed a bit to help learning

    # Device traffic model
    per_device_rate_range_bps: Tuple[float, float] = (15000.0, 20000.0)
    initial_data_bits_per_device: float = 300_000.0

    # Reward shaping
    reward_data_scale: float = 1.0 / 800.0
    reward_link_bonus: float = 6.0
    reward_distance_gain: float = 2.0
    reward_sinr_gain: float = 2.5

    penalty_energy_scale: float = 1.0
    penalty_out_of_bounds: float = 25.0
    penalty_collision: float = 35.0
    penalty_no_link: float = 1.0
    penalty_no_progress: float = 1.5

    # Observation encoding
    obs_clip_distance: float = 15.0

    # Fixed obstacles for 3D+5O
    obstacles: List[Obstacle3D] = field(default_factory=lambda: [
        Obstacle3D(x=12.0, y=12.0, width=4.0, length=4.0, height=4.0),
        Obstacle3D(x=20.0, y=20.0, width=4.0, length=4.0, height=7.0),
        Obstacle3D(x=26.0, y=16.0, width=4.0, length=4.0, height=2.5),
        Obstacle3D(x=31.0, y=25.0, width=4.0, length=4.0, height=5.0),
        Obstacle3D(x=36.0, y=26.0, width=4.0, length=4.0, height=5.0),
    ])


@dataclass
class TrainConfig:
    # close to Chapter 4, but epsilon is allowed to decay from a higher value
    learning_rate: float = 0.0004
    gamma: float = 0.99

    epsilon_start: float = 0.20
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995

    target_update_freq: int = 100
    minibatch_size: int = 32
    iterations: int = 150   # tăng lên 80000 sau khi test ổn

    replay_capacity: int = 50000
    hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    warmup_steps: int = 500
    train_every: int = 1
    seed: int = 42
    device: str = "cpu"

    output_dir: Path = Path("outputs")


def build_action_table(step_xy: float, step_z: float):
    """
    IMPORTANT:
    Remove (0,0,0) so the agent cannot learn the degenerate 'stand still forever' policy.
    """
    values_xy = (-step_xy, 0.0, step_xy)
    values_z = (-step_z, 0.0, step_z)
    actions = list(itertools.product(values_xy, values_xy, values_z))
    actions = [a for a in actions if not (a[0] == 0.0 and a[1] == 0.0 and a[2] == 0.0)]
    return actions
