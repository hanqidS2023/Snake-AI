import time
import random

import torch
from sb3_contrib import MaskablePPO

from cnn_env import SnakeEnv
from settings import *

MODEL_PATH = 'ppo_snake_model/maskableppo_snake_model'
NUM_EPISODE = 10

SHOW = True
ROUND_DELAY = 5

seed = random.randint(-INF, INF)
print(f"Using seed = {seed} for testing.")

env = SnakeEnv(seed=seed, show=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False

    num_step = 0
    info = None

    sum_step_reward = 0

    print(
        f"=================== Episode {episode + 1} ==================")
    while not done:
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        num_step += 1
        obs, reward, done, info = env.step(action)

        if done:
            last_action = ["UP", "DOWN", "LEFT", "RIGHT"][action]

        elif info["food_obtained"]:
            sum_step_reward = 0

        else:
            sum_step_reward += reward

        episode_reward += reward
        if SHOW:
            env.render()
            time.sleep(0.05)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score

    snake_size = info["snake_size"] + 1
    print(
        f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if SHOW:
        time.sleep(ROUND_DELAY)

env.close()