import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import numpy as np

# GPU optimization for faster matrix ops
torch.set_float32_matmul_precision('high')


# -------------------- ENVIRONMENT SETUP --------------------
def make_env():
    env = gym.make("highway-v0")
    env.unwrapped.configure({
        "observation": {"type": "Kinematics"},
        "policy_frequency": 15,
        "duration": 50,
        "lanes_count": 3,
        "vehicles_count": 20,
        "simulation_frequency": 25,
        "reward_speed_range": [20, 30],
        "collision_reward": -15,
        "controlled_vehicles": 1,
    })

    # -------- Custom Reward Function --------
    def reward_function(env, action):
        reward = 0.0
        ego = env.vehicle
        front_vehicle = ego.front_vehicle

        # 1️⃣ Base reward for staying alive
        reward += 1.0

        # 2️⃣ Smooth and safe driving reward
        reward += 0.05 * (ego.speed / 30)

        # 3️⃣ Maintain safe following distance
        if front_vehicle:
            dist = front_vehicle.position[0] - ego.position[0]
            if 10 < dist < 25:
                reward += 0.5    # good distance
            elif dist < 5:
                reward -= 0.5    # tailgating

        # 4️⃣ Reward for safe overtaking attempt
        if abs(ego.target_lane_index[2] - ego.lane_index[2]) > 0 and not ego.crashed:
            reward += 0.5

        # 5️⃣ Big reward for successful safe overtake
        if front_vehicle and ego.position[0] > front_vehicle.position[0] and not ego.crashed:
            reward += 3.0

        # 6️⃣ Heavy penalty for collision
        if ego.crashed:
            reward -= 15

        return reward

    env.define_reward = reward_function
    return env


# -------------------- ENV WRAPPING --------------------
env = DummyVecEnv([make_env])

# -------------------- MODEL SETUP --------------------
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=120000,
    batch_size=256,
    gamma=0.95,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    target_update_interval=500,
    train_freq=1,
    gradient_steps=2,
    device="cuda"
)

# -------------------- TRAINING --------------------
print("🚗 Training Realistic Safe-Overtake Agent (GPU accelerated)...")
model.learn(total_timesteps=150000)
model.save("rl_overtake_safe_realistic_v2.zip")
print("✅ Training complete! Saved as rl_overtake_safe_realistic_v2.zip")
