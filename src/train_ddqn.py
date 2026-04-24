import time
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from config import ScenarioConfig, TrainConfig
from envs.uav_iot_env import UAVIoTEnv
from agents.ddqn_agent import DDQNAgent
from utils.visualization import plot_3d_trajectory, plot_2d_topview, plot_convergence


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(env: UAVIoTEnv, agent: DDQNAgent, seed=None):
    if seed is None:
        state, info = env.reset()
    else:
        state, info = env.reset(seed=seed)

    done = False
    total_reward = 0.0

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.select_action(state)
        state, reward, done, _, info = env.step(action)
        total_reward += reward

    agent.epsilon = old_eps
    return total_reward, info


def main():
    scenario_cfg = ScenarioConfig()
    train_cfg = TrainConfig()
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Different random seed each run
    run_seed = int(time.time()) % 100000
    set_seed(run_seed)
    print("Run seed =", run_seed)

    env = UAVIoTEnv(scenario_cfg, seed=run_seed)

    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=train_cfg.learning_rate,
        gamma=train_cfg.gamma,
        epsilon_start=train_cfg.epsilon_start,
        epsilon_min=train_cfg.epsilon_min,
        epsilon_decay=train_cfg.epsilon_decay,
        replay_capacity=train_cfg.replay_capacity,
        target_update_freq=train_cfg.target_update_freq,
        hidden_dims=train_cfg.hidden_dims,
        device=train_cfg.device,
    )

    episode_rewards = []
    episode_losses = []
    best_eval_reward = -1e18

    progress = tqdm(range(train_cfg.iterations), desc="Training DDQN")
    for episode in progress:
        state, info = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            agent.replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward

            if len(agent.replay_buffer) >= max(train_cfg.minibatch_size, train_cfg.warmup_steps):
                if agent.global_step % train_cfg.train_every == 0:
                    loss = agent.update(train_cfg.minibatch_size)
                    if loss is not None:
                        losses.append(loss)
                else:
                    agent.global_step += 1

        episode_rewards.append(total_reward)
        episode_losses.append(float(np.mean(losses)) if losses else np.nan)

        # random evaluation rollout
        eval_reward, _ = evaluate_policy(env, agent, seed=None)
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            agent.save(str(train_cfg.output_dir / "best_ddqn.pt"))

        progress.set_postfix(
            reward=f"{total_reward:.2f}",
            eval=f"{eval_reward:.2f}",
            eps=f"{agent.epsilon:.4f}",
        )

    history_df = pd.DataFrame({
        "episode": np.arange(1, len(episode_rewards) + 1),
        "reward": episode_rewards,
        "loss": episode_losses,
    })
    history_df.to_csv(train_cfg.output_dir / "training_history.csv", index=False)

    # Final rollout using best model
    agent.load(str(train_cfg.output_dir / "best_ddqn.pt"))
    final_reward, final_info = evaluate_policy(env, agent, seed=None)

    print("\nBest eval reward:", best_eval_reward)
    print("Final eval reward:", final_reward)
    print("Collected bits:", final_info["collected_bits"])
    print("Remaining bits:", final_info["device_remaining_bits"])
    print("Battery:", final_info["battery"])

    traj = final_info["trajectory"]
    if len(traj) > 0:
        print("Trajectory length:", len(traj))
        print("Unique XY positions:", len(np.unique(traj[:, :2], axis=0)))
        print("Unique XYZ positions:", len(np.unique(traj, axis=0)))

    if len(final_info["link_history"]) > 0:
        active_steps = np.sum(np.any(final_info["link_history"] > 0.5, axis=1))
        print("Active link steps:", active_steps, "/", len(final_info["link_history"]))

    # Figure 1: 3D
    plot_3d_trajectory(
        scenario_cfg,
        trajectory=final_info["trajectory"],
        device_positions=final_info["device_positions"],
        jammer_positions=final_info["jammer_positions"],
        save_path=train_cfg.output_dir / "trajectory_3d.png",
    )

    # Figure 2: 2D
    plot_2d_topview(
        scenario_cfg,
        trajectory=final_info["trajectory"],
        device_positions=final_info["device_positions"],
        jammer_positions=final_info["jammer_positions"],
        link_history=final_info["link_history"],
        save_path=train_cfg.output_dir / "trajectory_2d.png",
    )

    # Figure 3: convergence
    plot_convergence(
        {"DDQN": episode_rewards},
        save_path=train_cfg.output_dir / "convergence.png",
    )

    summary_df = pd.DataFrame([{
        "run_seed": run_seed,
        "best_eval_reward": best_eval_reward,
        "final_eval_reward": final_reward,
        "final_collected_bits": final_info["collected_bits"],
        "final_battery": final_info["battery"],
    }])
    summary_df.to_csv(train_cfg.output_dir / "run_summary.csv", index=False)
    print("Outputs saved in ./outputs")
    plt.show()
if __name__ == "__main__":
    main()
    
    
