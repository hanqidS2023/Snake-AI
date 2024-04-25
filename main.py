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
        self.best_individual = None
        self.average_score = 0

    def calculate_fitness(self):
        game = Game([self.genes])
        self.score, self.steps, self.seed = game.play()
        self.fitness = self.score + (1.0 / self.steps)


class GA:
    def __init__(self):
        self.pop_size = POP_SIZE
        self.child_size = CHILD_SIZE
        self.genes_len = GENES_LEN
        self.mutation_rate = MUTATION_RATE
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
        gene1[:l+1], gene2[:l+1] = c2_genes[:l+1], gene2[:l+1]

    def mutate(self, c_genes):
        mutation_array = np.random.random(c_genes.shape) < self.mutation_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= 0.2
        c_genes[mutation_array] += mutation[mutation_array]

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
        total_score = 0
        for individual in self.population:
            individual.calculate_fitness()
            total_score += individual.score
        self.average_score = total_score / len(self.population)

        self.population = self.select(self.pop_size)
        self.best_individual = self.population[0]
        random.shuffle(self.population)

        children = []
        while (len(children) < self.child_size):
            p1, p2 = self.select(2)
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()
            self.crossover(c1_genes, c2_genes)
            self.mutate(c1_genes)
            self.mutate(c2_genes)
            c1, c2 = Individual(c1_genes), Individual(c2_genes)
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

    def save_all(self):
        for individual in self.population:
            individual.calculate_fitness()
        population = self.select(self.pop_size)
        for i in range(len(population)):
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "w") as f:
                for gene in self.population[i].genes:
                    f.write(str(gene) + " ")


def train_ga(show=True, load=False):
    ga = GA()
    record = 0
    generation = 0

    if load:
        ga.load_first_generation()
    else:
        ga.generate_first_generation()

    while True:
        generation += 1
        ga.evolve()

        best_score = ga.best_individual.score

        # Save the best individual if it surpasses the previous record
        if best_score > record:
            record = best_score
            ga.save_best()

        if show:
            genes = ga.best_individual.genes
            seed = ga.best_individual.seed
            game = Game(show=True, genes_list=[genes], seed=seed)
            game.play()

        # if generation % 20 == 0:
        #     ga.save_all()

        print(
            f"Generation: {generation}, Best Score: {best_score}, Record: {record}, Average: {ga.average_score}")


def run_multiple_tasks(window_num, show, load):
    with mp.Pool(window_num) as pool:
        # Run multiple instances with the specified parameters
        results = pool.starmap(
            train_ga, [(show, load) for _ in range(window_num)]
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
    if args.l:
        load = True
    if args.p and args.p > 1:
        window = args.p

    run_multiple_tasks(window, show, load)
