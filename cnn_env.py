import gym
import math
from gym import spaces
import numpy as np
import random
from settings import *
from snake_game_cnn import Game


class SnakeEnv(gym.Env):
    def __init__(self, seed=0, rows=ROWS, cols=COLS, show=False):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.game = Game(seed=seed, show=show, rows=rows, cols=cols)
        self.game.reset()

        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(60, 60, 3),
            dtype=np.uint8
        )

        self.board_size = rows * cols
        self.max_size = self.board_size - 1
        self.end = False

    def reset(self):
        self.game.reset()

        self.end = False

        obs = self.get_observation()
        return obs
    
    def render(self):
        if self.game.show:
            self.game.render()

    def get_observation(self):
        # Generate image for CNN
        obs = np.zeros((self.rows, self.cols), dtype=np.uint8)
        obs = np.stack((obs, obs, obs), axis=-1)
        obs[tuple(self.game.snake.body[0])] = [0, 255, 0]  # head to green
        obs[tuple(self.game.snake.body[-1])] = [0, 0, 255]  # tail to blue
        obs[tuple(self.game.food)] = [255, 0, 0]  # food to red

        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)
        return obs

    def step(self, action):
        dead, info_map = self.game.step(action)
        self.end = dead

        obs = self.get_observation()

        reward = 0.0
        # self.reward_step_counter += 1

        if self.end: 
            # dead, penalty is based on snake size.
            reward = -math.pow(self.max_size - info_map["snake_size"], 1)
            # reward = - math.pow(self.max_size, (self.board_size - info_map["snake_size"]) / self.max_size)
            reward *= 0.1
            return obs, reward, self.end, info_map

        elif info_map["food_eaten"]: 
            reward = info_map["snake_size"] / self.board_size

        else:
            old_euclidean_distance = np.linalg.norm(info_map["prev_snake_head_pos"] - info_map["food_pos"])
            new_euclidean_distance = np.linalg.norm(info_map["snake_head_pos"] - info_map["food_pos"])
            if old_euclidean_distance < new_euclidean_distance:
                reward = 1 / info_map["snake_size"]
            else:
                reward = - 1 / info_map["snake_size"]
            reward *= 0.1

        return obs, reward, self.end, info_map
    
    def action_mask(self, action):
        snake_direction = self.game.snake.direction
        head = (self.game.snake.body[0][0], self.game.snake.body[0][1])
        
        if action == 0: # up
            if snake_direction == DIRECTIONS[1]: 
                return False
            new_head = (head[0] + DIRECTIONS[0][0], head[1] + DIRECTIONS[0][1])
            if (new_head[0] < 0 or new_head[0] >= self.rows or head[1] < 0 or head[1] >= self.cols or head in self.game.snake.body[:-1]):
                return False
        if action == 1: # down
            if snake_direction == DIRECTIONS[0]: 
                return False
            new_head = (head[0] + DIRECTIONS[1][0], head[1] + DIRECTIONS[1][1])
            if (new_head[0] < 0 or new_head[0] >= self.rows or head[1] < 0 or head[1] >= self.cols or head in self.game.snake.body[:-1]):
                return False
        if action == 2: # left
            if snake_direction == DIRECTIONS[3]: 
                return False
            new_head = (head[0] + DIRECTIONS[2][0], head[1] + DIRECTIONS[2][1])
            if (new_head[0] < 0 or new_head[0] >= self.rows or head[1] < 0 or head[1] >= self.cols or head in self.game.snake.body[:-1]):
                return False
        if action == 3: # right
            if snake_direction == DIRECTIONS[2]: 
                return False
            new_head = (head[0] + DIRECTIONS[3][0], head[1] + DIRECTIONS[3][1])
            if (new_head[0] < 0 or new_head[0] >= self.rows or head[1] < 0 or head[1] >= self.cols or head in self.game.snake.body[:-1]):
                return False
        
        return True

    def action_masks(self):
        return np.array([[self.action_mask(action) for action in range(self.action_space.n)]])
        
            

    # def render(self, mode="human"):
    #     pg.init()
    #     screen = pg.display.set_mode((self.cols * GRID_SIZE, self.rows * GRID_SIZE))
    #     screen.fill((0, 0, 0))
    #     for segment in self.snake.body:
    #         pg.draw.rect(screen, GREEN, pg.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    #     food_pos = self.food
    #     pg.draw.rect(screen, RED, pg.Rect(food_pos[0] * GRID_SIZE, food_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    #     pg.display.flip()

    # def close(self):
    #     pg.quit()

# Example usage with MaskablePPO
