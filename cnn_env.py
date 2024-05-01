import math

import gymnasium as gym
import numpy as np

from snake_game_cnn import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, seed=0, rows=10, cols=10, show=False, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, rows=rows, cols=cols, show=show)
        self.game.reset()
        
        self.rows = rows
        self.cols = cols
        self.grid_size = self.game.grid_size
        self.init_snake_size = len(self.game.snake.body)
        self.max_growth = self.grid_size - self.init_snake_size
        self.done = False
        self.limit_step = limit_step

        # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.action_space = gym.spaces.Discrete(4) 
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(60, 60, 3),
            dtype=np.uint8
        )

        # Set a threshold to prevent making circle
        if limit_step:
            self.step_limit = self.grid_size * 4
        else:
            self.step_limit = 1e9
        self.step_counter = 0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed_value = seed
            self.game = SnakeGame(seed=seed, rows=self.rows, cols=self.cols, show=self.show)
        self.game.reset()

        self.done = False
        self.step_counter = 0

        obs = self._generate_observation()
        reset_info = {}
        return obs, reset_info
    
    def step(self, action):
        self.done, info = self.game.step(action) 
        obs = self._generate_observation()

        reward = 0.0
        self.step_counter += 1
        truncated = False

        if info["snake_size"] == self.grid_size:
            reward = self.max_growth * 0.1
            self.done = True
            return obs, reward, self.done, truncated, info
        
        if self.limit_step and self.step_counter > self.step_limit:
            self.step_counter = 0
            self.done = True
            truncated = True
        
        if self.done:
            reward = - math.pow(self.max_growth, (self.max_growth - info["snake_size"]) / self.max_growth)         
            reward = reward * 0.1
            return obs, reward, self.done, truncated, info
          
        elif info["food_obtained"]:
            reward = info["snake_size"] / self.max_growth
            self.step_counter = 0
        
        else:
            old_euclidean_distance = np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"])
            new_euclidean_distance = np.linalg.norm(info["snake_head_pos"] - info["food_pos"])
            if new_euclidean_distance < old_euclidean_distance:
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        return obs, reward, self.done, truncated, info
    
    def render(self):
        self.game.render()

    def action_masks(self):
        return np.array([[self.is_action_valid(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def is_action_valid(self, action):
        current_direction = self.game.snake.direction
        row, col = self.game.snake.body[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in self.game.snake.body # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.rows
                or col < 0
                or col >= self.cols
            )
        else:
            game_over = (
                (row, col) in  self.game.snake.body[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.rows
                or col < 0
                or col >= self.cols
            )

        if game_over:
            return False
        else:
            return True

    def _generate_observation(self):
        obs = np.zeros((self.game.rows, self.game.cols), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake.body))] = np.linspace(200, 50, len(self.game.snake.body), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake.body[0])] = [0, 255, 0]
        obs[tuple(self.game.snake.body[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)

        return obs