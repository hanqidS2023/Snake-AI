import pygame as pg
import random
from nn import NN
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
        self.nn = NN(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, genes.copy())
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
    def __init__(self, genes_list, seed=None, show=False, rows=ROWS, cols=COLS) -> None:
        self.Y = rows
        self.X = cols
        self.show = show
        self.seed = seed if seed is not None else random.randint(-INF, INF)
        self.rand = random.Random(self.seed)

        # Create new snakes, both with 1 length.
        self.snakes = []
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        for genes in genes_list:
            head = self.rand.choice(board)
            direction = DIRECTIONS[self.rand.randint(0, 3)]
            self.snakes.append(Snake(head, direction, genes, self.X, self.Y))

        self.food = self.new_food(self.snakes)
        self.best_score = 0

        if show:
            pg.init()
            self.width = cols * GRID_SIZE
            self.height = rows * GRID_SIZE + BLANK_SIZE

            pg.display.set_caption(TITLE)
            self.screen = pg.display.set_mode((self.width, self.height))
            self.clock = pg.time.Clock()
            self.font_name = pg.font.match_font(FONT_NAME)

    def play(self):
        snakes_set = set(self.snakes)
        while snakes_set and self.food is not None:
            if self.show:
                self.draw()

            for snake in snakes_set:
                has_eat = snake.move(self.food)
                if has_eat:
                    self.food = self.new_food(self.snakes)
                    if self.food is None:
                        break
                if snake.score > self.best_score:
                    self.best_score = snake.score
            alive_snakes = [
                snake for snake in snakes_set if not snake.dead]
            snakes_set = set(alive_snakes)

        if len(self.snakes) > 1:
            score = [snake.score for snake in self.snakes]
            steps = [snake.steps for snake in self.snakes]
        else:
            score, steps = self.snakes[0].score, self.snakes[0].steps

        return score, steps, self.seed

    def new_food(self, snakes):
        """Find an empty grid to place food."""
        board = set([(x, y) for x in range(self.X) for y in range(self.Y)])
        for snake in snakes:
            if not snake.dead:
                for body in snake.body:
                    board.discard(body)

        if len(board) == 0:
            return None

        return self.rand.choice(list(board))

    def draw(self):
        self.screen.fill(BLACK)
        def get_xy(pos): return (pos[0] * GRID_SIZE,
                                 pos[1] * GRID_SIZE + BLANK_SIZE)

        num = 0
        for snake in self.snakes:
            if not snake.dead:
                num += 1
                x, y = get_xy(snake.body[0])
                pg.draw.rect(self.screen, WHITE1, pg.Rect(
                    x, y, GRID_SIZE, GRID_SIZE))

                for s in snake.body[1:]:
                    x, y = get_xy(s)
                    pg.draw.rect(self.screen, snake.color,
                                 pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

        x, y = get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))

        text = "snakes: " + str(num) + " best score: " + str(self.best_score)
        font = pg.font.Font(self.font_name, 20)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.midtop = ((self.width / 2, 5))
        self.screen.blit(text_surface, text_rect)

        x = self.width // GRID_SIZE
        y = (self.height - BLANK_SIZE) // GRID_SIZE + 1
        for i in range(0, x):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE),
                         (i * GRID_SIZE, (y - 1) * GRID_SIZE + BLANK_SIZE), 1)
        for i in range(0, y):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE),
                         (self.width, i * GRID_SIZE + BLANK_SIZE), 1)

        pg.display.flip()


def play_best(score):
    genes_pth = os.path.join("genes", "best", str(score))
    with open(genes_pth, "r") as f:
        genes = np.array(list(map(float, f.read().split())))

    seed_pth = os.path.join("seed", str(score))
    with open(seed_pth, "r") as f:
        seed = int(f.read())

    game = Game(show=True, genes_list=[genes], seed=seed)
    game.play()

if __name__ == '__main__':
    play_best(74)
