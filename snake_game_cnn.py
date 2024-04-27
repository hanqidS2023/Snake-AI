import time
import gym
from gym import spaces
import numpy as np
import random
import pygame as pg
from settings import *


class Snake:
    def __init__(self, head, direction, board_x, board_y):
        self.body = [head]
        self.direction = direction
        self.score = 0
        self.dead = False
        self.board_x = board_x
        self.board_y = board_y
        self.uniq = [0] * board_x * board_y
        self.steps = 0

    def move(self, action, food):
        self.steps += 1

        self.direction = DIRECTIONS[action]
        head = (self.body[0][0] + self.direction[0],
                self.body[0][1] + self.direction[1])

        eat = False
        if (head[0] < 0 or head[0] >= self.board_x or head[1] < 0 or head[1] >= self.board_y or head in self.body[:-1]):
            self.dead = True
        else:
            self.body.insert(0, head)
            if head == food:
                self.score += 1
                eat = True
            else:
                self.body.pop()
                if (head, food) not in self.uniq:
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:
                    self.dead = True
        return eat


class Game:
    def __init__(self, seed=None, show=False, rows=ROWS, cols=COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.show = show
        self.seed = seed if seed is not None else random.randint(-INF, INF)
        self.rand = random.Random(self.seed)

        self.snake = None
        self.score = 0
        self.food = None
        self.reset()

    def reset(self):
        head = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        direction = random.choice(DIRECTIONS)
        # Second segment is calculated from the head and direction
        tail = (head[0] - direction[0], head[1] - direction[1])
        tail = (min(max(tail[0], 0), self.rows - 1), min(max(tail[1], 0), self.cols - 1))
        self.snake = Snake(head, direction, self.rows, self.cols)
        self.snake.body.append(tail)
        self.food = self.new_food()
        self.score = 0

    def new_food(self):
        available = [(x, y) for x in range(self.cols)
                     for y in range(self.rows) if (x, y) not in self.snake.body]
        if not available:
            return None
        return random.choice(available)

    def step(self, action):
        food_eaten = self.snake.move(action, self.food)
        if food_eaten:
            self.score += 1
            self.food = self.new_food()

        dead = self.snake.dead
        info_map = {
            "snake_size": len(self.snake.body),
            "snake_head_pos": np.array(self.snake.body[0]),
            "prev_snake_head_pos": np.array(self.snake.body[1]),
            "food_pos": np.array(self.food),
            "food_eaten": food_eaten
        }

        return dead, info_map
    
    def render(self):
        pg.init()
        self.width = self.cols * GRID_SIZE
        self.height = self.rows * GRID_SIZE + BLANK_SIZE

        pg.display.set_caption(TITLE)
        self.screen = pg.display.set_mode((self.width, self.height))
        self.clock = pg.time.Clock()
        self.font_name = pg.font.match_font(FONT_NAME)
        
        if not self.snake.dead:
            print("Game loop running")
            print("Snake position:", self.snake.body[0])
            print("Snake direction:", self.snake.direction)
            self.draw()
            time.sleep(0.1) 

    def draw(self):
        self.screen.fill(BLACK)

        def get_xy(pos): return (pos[0] * GRID_SIZE,
                                 pos[1] * GRID_SIZE + BLANK_SIZE)

        num = 0
        if not self.snake.dead:
            num += 1
            x, y = get_xy(self.snake.body[0])
            pg.draw.rect(self.screen, WHITE1, pg.Rect(
                x, y, GRID_SIZE, GRID_SIZE))

            for s in self.snake.body[1:]:
                x, y = get_xy(s)
                pg.draw.rect(self.screen, GREEN,
                                pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

        x, y = get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))


        x = self.width // GRID_SIZE
        y = (self.height - BLANK_SIZE) // GRID_SIZE + 1
        for i in range(0, x):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE),
                         (i * GRID_SIZE, (y - 1) * GRID_SIZE + BLANK_SIZE), 1)
        for i in range(0, y):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE),
                         (self.width, i * GRID_SIZE + BLANK_SIZE), 1)

        pg.display.flip()
