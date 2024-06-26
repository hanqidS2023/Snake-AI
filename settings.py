TITLE = "Snake-MLP"
GRID_SIZE = 40
BLANK_SIZE = 40
ROWS = 10
COLS = 10
RED = (255, 0, 0)
GREEN = (0, 255, 0)
INF = 999999999

N_INPUT = 32
N_HIDDEN1 = 20
N_HIDDEN2 = 12
N_OUTPUT = 4
GENES_LEN = N_INPUT * N_HIDDEN1 + N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 * N_OUTPUT + N_HIDDEN1 + N_HIDDEN2 + N_OUTPUT 
POP_SIZE = 500
CHILD_SIZE = 400
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
MUTATION_RATE = 0.1
