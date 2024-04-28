import math
import random
import time
from sb3_contrib import MaskablePPO
import gym
import numpy as np
import os

import torch
from settings import *
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


from cnn_env import SnakeEnv

# Class for training the Snake game with MaskablePPO and CNN


class MaskableSnakeTrainer:
    def __init__(self, env, model_save_path, total_timesteps=1000000, learning_rate=3e-4, tensorboard_log='./tensorboard_logs'):
        self.env = env
        self.model_save_path = model_save_path
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate

        # Initialize the MaskablePPO model with a CNN policy
        self.model = MaskablePPO(
            "CnnPolicy",
            self.env,
            batch_size=1024*8,
            device="cuda",
            learning_rate=self.learning_rate,
            n_epochs=5,
            gamma=0.95,
            verbose=1,
            tensorboard_log=tensorboard_log
        )

    def train(self, log_dir):
        print("Training MaskablePPO with CNN...")
        checkpoint_freq = 10000
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq, save_path=log_dir, name_prefix="snake")
        self.model.learn(total_timesteps=self.total_timesteps,
                         callback=[checkpoint_callback])
        self.env.close()
        print("Training complete.")

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.model.save(os.path.join(
            self.model_save_path, "maskableppo_snake_model"))
        print(
            f"Model saved at {os.path.join(self.model_save_path, 'maskableppo_snake_model')}")

    def load_model(self):
        MODEL_PATH = 'ppo_snake_model/maskableppo_snake_model'
        NUM_EPISODE = 10

        RENDER = True
        FRAME_DELAY = 0.05  # 0.01 fast, 0.05 slow
        ROUND_DELAY = 5

        seed = random.randint(-INF, INF)
        print(f"Using seed = {seed} for testing.")

        env = SnakeEnv(seed=seed, show=RENDER)

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

            retry_limit = 9
            print(
                f"=================== Episode {episode + 1} ==================")
            while not done:
                action, _ = model.predict(obs, action_masks=env.action_masks())
                prev_mask = env.action_masks()
                prev_direction = env.game.snake.direction
                num_step += 1
                obs, reward, done, info = env.step(action)

                if done:
                    if info["snake_size"] == env.board_size:
                        print(
                            f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
                    else:
                        last_action = ["UP", "DOWN", "LEFT", "RIGHT"][action]
                        print(
                            f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

                elif info["food_eaten"]:
                    print(
                        f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
                    sum_step_reward = 0

                else:
                    sum_step_reward += reward

                episode_reward += reward
                if RENDER:
                    env.render()
                    time.sleep(FRAME_DELAY)

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
            if RENDER:
                time.sleep(ROUND_DELAY)

        env.close()


def make_env(seed=0, show=False):
    def _init():
        env = SnakeEnv(seed=seed, show=show)
        return env
    return _init


def sine_wave_schedule(min_lr: float, max_lr: float, cycles: int = 1):
    def func(progress_remaining: float) -> float:
        # Sine function to oscillate the learning rate between min and max
        amplitude = (max_lr - min_lr) / 2
        mid_point = (max_lr + min_lr) / 2
        return amplitude * math.sin(2 * math.pi * cycles * (1 - progress_remaining)) + mid_point
    return func


if __name__ == '__main__':
    # Create 6 parallel environments
    num_envs = 6

    seed_set = set()
    for _ in range(num_envs):
        seed_set.add(random.randint(-INF, INF))

    env = SubprocVecEnv([make_env(seed=seed, show=True) for seed in seed_set])

    learning_rate_schedule = sine_wave_schedule(3e-5, 3e-4)

    log_dir = './tensorboard_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the trainer
    trainer = MaskableSnakeTrainer(
        env,
        model_save_path="ppo_snake_model",
        total_timesteps=100000000,
        learning_rate=learning_rate_schedule,
        tensorboard_log=log_dir
    )

    # Train the model
    trainer.train(log_dir)

    # Save the model after training
    trainer.save_model()

    # trainer.load_model()

    # if torch.backends.mps.is_available():
    #     print('haha')
