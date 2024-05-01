import sys
import time
import pygame as pg
import random
from mlp import MLP
from settings import *
import os
import numpy as np


class Snake:
    def __init__(self, head, direction, genes, board_x, board_y):
        self.body = [head]
        self.direction = direction
        self.score = 0
        self.steps = 0
        self.reward = 0
        self.dead = False
        self.board_x = board_x
        self.board_y = board_y
        self.uniq = [0] * board_x * board_y
        self.nn = MLP(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, genes.copy())
        self.color = GREEN

    def get_state(self, food):
        # Head direction
        i = DIRECTIONS.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        # Tail direction
        if len(self.body) == 1:
            tail_direction = self.direction
        else:
            tail_direction = (
                self.body[-2][0] - self.body[-1][0], self.body[-2][1] - self.body[-1][1])
        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0
        state = head_dir + tail_dir

        # Vision of 8 directions.
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1],
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]

        for dir in dirs:
            x = self.body[0][0] + dir[0]
            y = self.body[0][1] + dir[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            while x >= 0 and x < self.board_x and y >= 0 and y < self.board_y:
                if (x, y) == food:
                    see_food = 1.0
                elif (x, y) in self.body:
                    see_self = 1.0
                dis += 1
                x += dir[0]
                y += dir[1]
            state += [1.0/dis, see_food, see_self]
        return state

    def move(self, food):
        self.steps += 1
        state = self.get_state(food)
        action = self.nn.predict(state)

        self.direction = DIRECTIONS[action]
        head = (self.body[0][0] + self.direction[0],
                self.body[0][1] + self.direction[1])
        row, col = head

        food_obtained = False
        done = (
            (row, col) in self.body
            or row < 0
            or row >= ROWS
            or col < 0
            or col >= COLS
        )
        if done:
            self.dead = True
        else:
            self.body.insert(0, head)
            if head == food:
                self.score += 10
                food_obtained = True
            else:
                self.body.pop()
                if (head, food) not in self.uniq:
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:
                    self.dead = True
        return food_obtained


class Game:
    def __init__(self, genes, seed=None, show=False, rows=ROWS, cols=COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.show = show
        self.seed = seed if seed is not None else random.randint(-INF, INF)
        self.rand = random.Random(self.seed)
        self.border_size = 20

        board = [(x, y) for x in range(self.rows) for y in range(self.cols)]
        head = self.rand.choice(board)
        direction = DIRECTIONS[self.rand.randint(0, 3)]
        self.snake = Snake(head, direction, genes, self.rows, self.cols)

        self.food = self.new_food()
        self.best_score = 0

        if show:
            pg.init()
            self.width = self.rows * GRID_SIZE
            self.height = self.cols * GRID_SIZE
            self.display_width = self.width + 2 * self.border_size
            self.display_height = self.height + 2 * self.border_size + 40
            self.font = pg.font.Font(None, 36)

            pg.display.set_caption(TITLE)
            self.screen = pg.display.set_mode((self.display_width, self.display_height))
            self.clock = pg.time.Clock()
        else:
            self.screen = None
            self.font = None

    def play(self):
        running = True
        while not self.snake.dead and running:
            if self.show:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False

            if self.show:
                self.draw()
                time.sleep(0.05)

            food_obtained = self.snake.move(self.food)
            if food_obtained:
                self.food = self.new_food()
                if self.food is None:
                    break
            if self.snake.score > self.best_score:
                self.best_score = self.snake.score

        score, steps = self.snake.score, self.snake.steps

        return score, steps, self.seed

    def new_food(self):
        board = set([(x, y) for x in range(self.rows)
                    for y in range(self.cols)])
        if not self.snake.dead:
            for body in self.snake.body:
                board.discard(body)

        if len(board) == 0:
            return None

        return self.rand.choice(list(board))

    def draw_snake(self):
        # draw head
        x, y = self.get_xy(self.snake.body[0])
        pg.draw.rect(self.screen, (220, 220, 220),
                     pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

        # draw body
        color_list = np.linspace(200, 50, len(self.snake.body), dtype=np.uint8)
        i = 0
        for s in self.snake.body[1:]:
            x, y = self.get_xy(s)
            pg.draw.rect(self.screen, (0, color_list[i], 0),
                         pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            i += 1

        # draw tail
        x, y = self.get_xy(
            self.snake.body[len(self.snake.body)-1])
        pg.draw.rect(self.screen, (0, 0, 255),
                     pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

    def draw_food(self):
        x, y = self.get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

    def get_xy(self, pos): 
        return (pos[0] * GRID_SIZE + BLANK_SIZE, pos[1] * GRID_SIZE + BLANK_SIZE)
    
    def draw_score(self):
        score_text = self.font.render(
            f"Score: {self.snake.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.border_size,
                         self.height + 2 * self.border_size))

    def draw(self):
        self.screen.fill((0, 0, 0))

        if not self.snake.dead:
            self.draw_snake()
            
        self.draw_food()
        
        self.draw_score()

        pg.display.flip()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()


def play_best(score):
    genes_pth = os.path.join("genes", "best", str(score))
    with open(genes_pth, "r") as f:
        genes = np.array(list(map(float, f.read().split())))

    seed_pth = os.path.join("seed", str(score))
    with open(seed_pth, "r") as f:
        seed = int(f.read())

    game = Game(show=True, genes=genes, seed=seed)
    game.play()

if __name__ == '__main__':
    play_best(74)
