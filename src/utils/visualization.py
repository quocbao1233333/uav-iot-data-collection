from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _cuboid_faces(x, y, z, dx, dy, dz):
    p = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz],
    ])
    return [
        [p[0], p[1], p[2], p[3]],
        [p[4], p[5], p[6], p[7]],
        [p[0], p[1], p[5], p[4]],
        [p[1], p[2], p[6], p[5]],
        [p[2], p[3], p[7], p[6]],
        [p[3], p[0], p[4], p[7]],
    ]


def plot_3d_trajectory(cfg, trajectory, device_positions, jammer_positions, save_path=None):
    fig = plt.figure("3D UAV Trajectory")
    ax = fig.add_subplot(111, projection="3d")

    for obs in cfg.obstacles:
        faces = _cuboid_faces(obs.x, obs.y, 0.0, obs.width, obs.length, obs.height)
        poly = Poly3DCollection(faces, facecolor="lightgray", edgecolor="gray", alpha=0.5)
        ax.add_collection3d(poly)

    traj = np.asarray(trajectory)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="black", linewidth=2.0)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=50, c="black", label="UAV start")
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker=">", s=15, c="tab:orange")

    dev = np.asarray(device_positions)
    jam = np.asarray(jammer_positions)

    ax.scatter(dev[:, 0], dev[:, 1], dev[:, 2], marker="o", s=70, c="green", label="IoT devices")
    ax.scatter(jam[:, 0], jam[:, 1], jam[:, 2], marker="s", s=70, c="red", label="Jammers")

    ax.set_xlim(0, cfg.x_max)
    ax.set_ylim(0, cfg.y_max)
    ax.set_zlim(0, cfg.z_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-62)
    ax.legend(loc="upper left")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def plot_2d_topview(cfg, trajectory, device_positions, jammer_positions, link_history=None, save_path=None):
    fig = plt.figure("2D Top View")
    ax = fig.add_subplot(111)

    obstacle_handles = []
    for obs in cfg.obstacles:
        rect = plt.Rectangle((obs.x, obs.y), obs.width, obs.length, color="lightgray", alpha=0.6)
        ax.add_patch(rect)

    traj = np.asarray(trajectory)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2.0)
        ax.scatter(traj[0, 0], traj[0, 1], s=80, c="black", label="UAV flight starting point")
        ax.scatter(traj[:, 0], traj[:, 1], marker="^", s=18, c="darkorange",
                   label="UAV movement direction / communication segment")

        if link_history is not None and len(link_history) == len(traj) - 1:
            colors = ["cyan", "pink", "green"]
            for t in range(len(link_history)):
                active = np.where(np.asarray(link_history[t]) > 0.5)[0]
                if len(active) > 0:
                    idx = int(active[0] % len(colors))
                    ax.plot(
                        traj[t:t+2, 0],
                        traj[t:t+2, 1],
                        color=colors[idx],
                        linewidth=3.0,
                        alpha=0.9,
                    )

    dev = np.asarray(device_positions)
    jam = np.asarray(jammer_positions)

    ax.scatter(jam[:, 0], jam[:, 1], marker="s", s=120, c="red", label="jammers")
    ax.scatter(dev[:, 0], dev[:, 1], marker="o", s=120, c="green", label="green IoT device")

    obstacle_patch = Patch(facecolor="lightgray", edgecolor="lightgray", alpha=0.6, label="obstacle")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(obstacle_patch)
    labels.append("obstacle")

    ax.set_xlim(0, cfg.x_max)
    ax.set_ylim(0, cfg.y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def moving_average(values, window=200):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_convergence(histories, save_path=None):
    fig = plt.figure("Convergence Performance")
    ax = fig.add_subplot(111)

    for name, values in histories.items():
        values = np.asarray(list(values), dtype=float)
        ax.plot(values, alpha=0.20, label=f"{name} raw")
        ax.plot(
            moving_average(values, window=min(200, max(10, len(values) // 20))),
            linewidth=2.5,
            label=name
        )

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Rewards")
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig

