import math
import random
import sys
import time
from sb3_contrib import MaskablePPO
import gym
import numpy as np
import os

import torch
from settings import *
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from cnn_env import SnakeEnv

# Class for training the Snake game with MaskablePPO and CNN


class MaskableSnakeTrainer:
    def __init__(self, env, model_save_path, total_timesteps=1000000, learning_rate=3e-4, clip_range=0.0125, tensorboard_log='./tensorboard_logs'):
        self.env = env
        self.model_save_path = model_save_path
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate

        # Initialize the MaskablePPO model with a CNN policy
        self.model = MaskablePPO(
            "CnnPolicy",
            self.env,
            device="cuda",
            learning_rate=self.learning_rate,
            clip_range=clip_range,
            n_epochs=4,
            gamma=0.95,
            verbose=1,
            tensorboard_log=tensorboard_log
        )

    def train(self, log_dir):
        print("Training MaskablePPO with CNN...")
        # Writing the training logs from stdout to a file
        original_stdout = sys.stdout
        log_file_path = os.path.join(self.model_save_path, "training_log.txt")
        checkpoint_freq = 10000
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq, save_path=log_dir, name_prefix="snake")
        with open(log_file_path, 'w') as log_file:
            sys.stdout = log_file

            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=[checkpoint_callback]
            )
            env.close()

        # Restore stdout
        sys.stdout = original_stdout
        self.env.close()
        print("Training complete.")

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.model.save(os.path.join(
            self.model_save_path, "maskableppo_snake_model_fixed_seed"))
        print(
            f"Model saved at {os.path.join(self.model_save_path, 'maskableppo_snake_model')}")


def make_env(seed=0, show=False):
    def _init():
        env = SnakeEnv(seed=seed, show=show)
        Monitor(env)
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
    num_envs = 32

    seed_set = set()
    for _ in range(num_envs):
        seed_set.add(random.randint(-INF, INF))

    env = SubprocVecEnv([make_env(seed=9999) for seed in seed_set])

    learning_rate_schedule = sine_wave_schedule(5e-4, 2.5e-6)
    clip_range_schedule = sine_wave_schedule(0.150, 0.025)

    log_dir = './tensorboard_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the trainer
    trainer = MaskableSnakeTrainer(
        env,
        model_save_path="ppo_snake_model",
        total_timesteps=20000000,
        learning_rate=learning_rate_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=log_dir
    )

    # Train the model
    trainer.train(log_dir)

    # Save the model after training
    trainer.save_model()
