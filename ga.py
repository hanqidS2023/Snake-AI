import random
import numpy as np
import argparse
from snake_game import Game
from settings import *
import multiprocessing as mp
import os


class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.seed = None
        self.fitness = 0.0
        self.average_score = 0

    def calculate_fitness(self):
        game = Game(self.genes)
        self.score, self.steps, self.seed = game.play()
        self.fitness = self.score + (1.0 / self.steps)


class GA:
    def __init__(self):
        self.pop_size = POP_SIZE
        self.child_size = CHILD_SIZE
        self.genes_len = GENES_LEN
        self.mutation_rate = MUTATION_RATE
        self.best_individual = None
        self.population = []

        self.generate_first_generation()

    def generate_first_generation(self):
        for _ in range(self.pop_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))

    def load_first_generation(self):
        for i in range(self.pop_size):
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                self.population.append(Individual(genes))

    def crossover(self, gene1, gene2):
        l = np.random.randint(0, self.genes_len)
        gene1[:l+1], gene2[:l+1] = gene2[:l+1], gene1[:l+1]

    def mutate(self, c_genes):
        mutation_index = np.random.random(c_genes.shape) < self.mutation_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_index] *= 0.2
        c_genes[mutation_index] += mutation[mutation_index]

    def elitism(self, size):
        population = sorted(
            self.population, key=lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def select(self, size):
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        for _ in range(size):
            pick = np.random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break
        return selection

    def evolve(self):
        # Calculate average score
        total_score = 0
        for individual in self.population:
            individual.calculate_fitness()
            total_score += individual.score
        self.average_score = total_score / len(self.population)

        # Apply elitism, get best individual
        self.population = self.elitism(self.pop_size - self.child_size)
        self.best_individual = self.population[0]
        random.shuffle(self.population)

        # Apply Roulette Wheel selection for each child
        children = []
        while (len(children) < self.child_size):
            p1, p2 = self.select(2)
            gene_1, gene_2 = p1.genes.copy(), p2.genes.copy()
            self.crossover(gene_1, gene_2)
            self.mutate(gene_1)
            self.mutate(gene_2)
            c1, c2 = Individual(gene_1), Individual(gene_2)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)

    def save_best(self):
        score = self.best_individual.score
        genes_pth = os.path.join("genes", "best", str(score))
        with open(genes_pth, "w") as f:
            for gene in self.best_individual.genes:
                f.write(str(gene) + " ")
        seed_pth = os.path.join("seed", str(score))
        with open(seed_pth, "w") as f:
            f.write(str(self.best_individual.seed))


def train_ga(show=True):
    ga = GA()
    record = 0
    generation = 0

    ga.generate_first_generation()

    while True:
        generation += 1
        ga.evolve()

        best_score = ga.best_individual.score
        
        if best_score == 99:
            break

        # Save the best individual if it surpasses the previous record
        if best_score > record:
            record = best_score
            ga.save_best()

        if show:
            genes = ga.best_individual.genes
            seed = ga.best_individual.seed
            game = Game(show=True, genes_list=[genes], seed=seed)
            game.play()

        print(
            f"Generation: {generation}, Best Score: {best_score}, Record: {record}, Average: {ga.average_score}")


def run_multiple_tasks(window_num, show):
    with mp.Pool(window_num) as pool:
        # Run multiple instances with the specified parameters
        results = pool.starmap(
            train_ga, [(show,) for _ in range(window_num)]
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', action="store_true")

    parser.add_argument('-s', action="store_true")

    parser.add_argument('-p', type=int, default=1)
    args = parser.parse_args()

    show, load, window = None, None, 1
    if args.s:
        show = True
    if args.p and args.p > 1:
        window = args.p

    run_multiple_tasks(window, show)
