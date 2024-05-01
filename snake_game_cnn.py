import os
import sys
import random

import numpy as np

import pygame as pg


class Snake:
    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.body = [(self.rows // 2 + i, self.cols // 2)
                     for i in range(1, -2, -1)]
        self.dead = False
        self.direction = 'DOWN'

    def update_direction(self, action):
        if action == 0:
            if self.direction != "DOWN":
                self.direction = "UP"
        elif action == 1:
            if self.direction != "RIGHT":
                self.direction = "LEFT"
        elif action == 2:
            if self.direction != "LEFT":
                self.direction = "RIGHT"
        elif action == 3:
            if self.direction != "UP":
                self.direction = "DOWN"

    def move(self, food):
        # Move snake based on current action.
        row, col = self.body[0]
        if self.direction == "UP":
            row -= 1
        elif self.direction == "DOWN":
            row += 1
        elif self.direction == "LEFT":
            col -= 1
        elif self.direction == "RIGHT":
            col += 1

        # Check if snake eats food.
        if (row, col) == food:
            food_obtained = True
        else:
            food_obtained = False
            self.body.pop()

        done = (
            (row, col) in self.body
            or row < 0
            or row >= self.rows
            or col < 0
            or col >= self.cols
        )

        if not done:
            self.body.insert(0, (row, col))

        return done, food_obtained


class SnakeGame:
    def __init__(self, seed=0, rows=10, cols=10, show=False):
        self.rows = rows
        self.cols = cols
        self.grid_size = self.rows * self.cols
        self.cell_size = 40
        self.width = self.rows * self.cell_size
        self.height = self.cols * self.cell_size

        self.border_size = 20
        self.display_width = self.width + 2 * self.border_size
        self.display_height = self.height + 2 * self.border_size + 40

        self.show = show
        if show:
            pg.init()
            pg.display.set_caption("Snake-CNN")
            self.screen = pg.display.set_mode(
                (self.display_width, self.display_height))
            self.font = pg.font.Font(None, 36)
        else:
            self.screen = None
            self.font = None

        self.snake = None
        self.score = 0
        self.food = None
        self.seed_value = seed

        self.rand = random.Random(seed)  # Set random seed.

        self.reset()

    def reset(self):
        self.snake = Snake()
        self.food = self.new_food()
        self.score = 0

    def step(self, action):
        # Update direction based on action.
        self.snake.update_direction(action)

        done, food_obtained = self.snake.move(self.food)

        # Add new food after snake movement completes.
        if food_obtained:
            self.food = self.new_food()
            self.score += 10

        info = {
            "snake_size": len(self.snake.body),
            "snake_head_pos": np.array(self.snake.body[0]),
            "prev_snake_head_pos": np.array(self.snake.body[1]),
            "food_pos": np.array(self.food),
            "food_obtained": food_obtained
        }

        return done, info

    def new_food(self):
        board = set([(x, y) for x in range(self.rows)
                    for y in range(self.cols)])
        for body in self.snake.body:
            board.discard(body)

        if len(board) == 0:
            return None

        return self.rand.choice(list(board))

    def draw_score(self):
        score_text = self.font.render(
            f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.border_size,
                         self.height + 2 * self.border_size))

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw snake
        self.draw_snake()

        r, c = self.food
        pg.draw.rect(self.screen, (255, 0, 0), (c * self.cell_size + self.border_size,
                                                r * self.cell_size + self.border_size, self.cell_size, self.cell_size))

        # Draw score
        self.draw_score()

        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

    def get_xy(self, pos, cell_size):
        return (pos[1] * cell_size + self.border_size,
                pos[0] * cell_size + self.border_size)

    def draw_snake(self):
        # draw head
        x, y = self.get_xy(self.snake.body[0], self.cell_size)
        pg.draw.rect(self.screen, (220, 220, 220), pg.Rect(
            x, y, self.cell_size, self.cell_size))

        # draw body
        color_list = np.linspace(200, 50, len(self.snake.body), dtype=np.uint8)
        i = 0
        for s in self.snake.body[1:]:
            x, y = self.get_xy(s, self.cell_size)
            pg.draw.rect(self.screen, (0, color_list[i], 0),
                         pg.Rect(x, y, self.cell_size, self.cell_size))
            i += 1

        # draw tail
        x, y = self.get_xy(
            self.snake.body[len(self.snake.body)-1], self.cell_size)
        pg.draw.rect(self.screen, (0, 0, 255),
                     pg.Rect(x, y, self.cell_size, self.cell_size))
