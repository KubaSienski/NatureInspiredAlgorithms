import random
import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 150
GENERATIONS = 500
DIMENSIONS = 30


class NSGAII:
    def __init__(self, population_size, generations, num_objectives, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.num_objectives = num_objectives
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self, bounds):
        """Initialize the population within the given bounds."""
        return np.random.uniform(bounds[:, 0], bounds[:, 1],
                                 (self.population_size, bounds.shape[0]))

    def evaluate_objectives(self, population, objective_functions):
        """Evaluate the objectives for all individuals."""
        return np.array([[obj(ind) for obj in objective_functions] for ind in population])

    def non_dominated_sorting(self, objectives):
        """Perform non-dominated sorting to classify individuals into Pareto fronts."""
        num_individuals = objectives.shape[0]
        domination_counts = np.zeros(num_individuals)
        dominated_sets = [[] for _ in range(num_individuals)]
        fronts = [[]]

        for i in range(num_individuals):
            for j in range(num_individuals):
                if self.dominates(objectives[i], objectives[j]):
                    dominated_sets[i].append(j)
                elif self.dominates(objectives[j], objectives[i]):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for individual in fronts[current_front]:
                for dominated in dominated_sets[individual]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front.append(dominated)
            fronts.append(next_front)
            current_front += 1

        return fronts[:-1]

    @staticmethod
    def dominates(ind1, ind2):
        """Check if ind1 dominates ind2."""
        return all(ind1 <= ind2) and any(ind1 < ind2)

    def crowding_distance(self, front, objectives):
        """Calculate the crowding distance for individuals in a front."""
        distances = np.zeros(len(front))
        for m in range(self.num_objectives):
            sorted_indices = np.argsort(objectives[front, m])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            min_value = objectives[front[sorted_indices[0]], m]
            max_value = objectives[front[sorted_indices[-1]], m]

            if max_value - min_value == 0:  # Avoid division by zero
                continue

            for i in range(1, len(front) - 1):
                distances[sorted_indices[i]] += (
                                                        objectives[front[sorted_indices[i + 1]], m] - objectives[
                                                    front[sorted_indices[i - 1]], m]
                                                ) / (max_value - min_value)

        return distances

    def tournament_selection(self, population, objectives,
                             fronts, crowding_distances):
        """Select individuals using tournament selection
            based on Pareto rank and crowding distance."""
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(len(population)), 2)
            if fronts[i] < fronts[j] or (fronts[i] == fronts[j] and
                                         crowding_distances[i] > crowding_distances[j]):
                selected.append(i)
            else:
                selected.append(j)
        return np.array(selected)

    def crossover(self, parent1, parent2):
        """Perform simulated binary crossover (SBX)."""
        eta = 20
        if random.random() < self.crossover_rate:
            child1 = np.empty_like(parent1)
            child2 = np.empty_like(parent1)
            for i in range(len(parent1)):
                if random.random() <= 0.5:
                    if abs(parent1[i] - parent2[i]) > 1e-14:
                        y1 = min(parent1[i], parent2[i])
                        y2 = max(parent1[i], parent2[i])
                        lower, upper = -5, 5
                        rand = random.random()
                        beta = 1.0 + (2.0 * (y1 - lower) / (y2 - y1))
                        alpha = 2.0 - beta ** -(eta + 1.0)
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        child1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper - y2) / (y2 - y1))
                        alpha = 2.0 - beta ** -(eta + 1.0)
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        child2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    else:
                        child1[i] = parent1[i]
                        child2[i] = parent2[i]
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            return np.clip(child1, -5, 5), np.clip(child2, -5, 5)
        else:
            return parent1, parent2

    def mutate(self, individual, bounds):
        """Mutate an individual by adding random noise."""
        mutation_rate = 1.0 / len(individual)
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                lower, upper = bounds[i, 0], bounds[i, 1]
                delta = (upper - lower)
                individual[i] += np.random.uniform(-0.1 * delta, 0.1 * delta)
                individual[i] = np.clip(individual[i], lower, upper)
        return individual

    def run(self, bounds, objective_functions):
        """Run the NSGA-II algorithm."""
        population = self.initialize_population(bounds)

        for _ in range(self.generations):
            # Combine parent and offspring populations
            offspring_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = population[i], population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                offspring_population.extend([self.mutate(child1, bounds),
                                             self.mutate(child2, bounds)])

            combined_population = np.vstack((population, np.array(offspring_population)))
            combined_objectives = self.evaluate_objectives(combined_population, objective_functions)

            fronts = self.non_dominated_sorting(combined_objectives)
            new_population = []
            new_objectives = []
            crowding_distances = []

            for front in fronts:
                if len(new_population) + len(front) > self.population_size:
                    distances = self.crowding_distance(front, combined_objectives)
                    sorted_indices = np.argsort(-distances)
                    front = [front[i] for i in sorted_indices[: self.population_size - len(new_population)]]

                new_population.extend(combined_population[front])
                new_objectives.extend(combined_objectives[front])
                crowding_distances.extend(self.crowding_distance(front, combined_objectives))

                if len(new_population) == self.population_size:
                    break

            population = np.array(new_population)

        return population, np.array(new_objectives)


# Define ZDT benchmark functions
def zdt1(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    ratio = f1 / g
    ratio = min(max(ratio, 0), 1)  # Clamp to [0, 1]
    f2 = g * (1 - np.sqrt(ratio))
    f1 = max(f1, 0)
    return [f1, f2]


def zdt2(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    ratio = f1 / g
    ratio = min(max(ratio, 0), 1)  # Clamp to [0, 1]
    f2 = g * (1 - (ratio) ** 2)
    f1 = max(f1, 0)
    return [f1, f2]


def zdt3(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    ratio = f1 / g
    ratio = min(max(ratio, 0), 1)  # Clamp to [0, 1]
    f2 = g * (1 - np.sqrt(ratio) - (ratio) * np.sin(10 * np.pi * f1))
    f1 = max(f1, 0)
    return [f1, f2]


def zdt4(x):
    f1 = x[0]
    g = 1 + 10 * (len(x) - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
    ratio = f1 / g
    ratio = min(max(ratio, 0), 1)  # Clamp to [0, 1]
    f2 = g * (1 - np.sqrt(ratio))
    f1 = max(f1, 0)
    return [f1, f2]


def zdt6(x):
    f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]) ** 6)
    base = max(np.sum(x[1:]) / (len(x) - 1), 0)  # Ensure non-negative base
    g = 1 + 9 * (base ** 0.25)
    ratio = f1 / g
    ratio = min(max(ratio, 0), 1)  # Clamp to [0, 1]
    f2 = g * (1 - (ratio) ** 2)
    f1 = max(f1, 0)
    return [f1, f2]


# Run benchmarks and save Pareto front plots
def run_and_plot():
    benchmarks = {
        "ZDT1": (zdt1, np.array([[0, 1]] * DIMENSIONS)),
        #"ZDT2": (zdt2, np.array([[0, 1]] * DIMENSIONS)),
        #"ZDT3": (zdt3, np.array([[0, 1]] * DIMENSIONS)),
        #"ZDT4": (zdt4, np.vstack(([0, 1], np.tile([-5, 5], (DIMENSIONS - 1, 1))))),
        #"ZDT6": (zdt6, np.array([[0, 1]] * DIMENSIONS)),
    }

    nsga2 = NSGAII(
        population_size=POP_SIZE, generations=GENERATIONS,
        num_objectives=2, crossover_rate=0.9, mutation_rate=0.1
    )

    for name, (objective_function, bounds) in benchmarks.items():
        population, objectives = nsga2.run(bounds, [
            lambda x: objective_function(x)[0], lambda x: objective_function(x)[1]
        ])

        # Plot the Pareto front
        plt.figure()
        plt.scatter(objectives[:, 0], objectives[:, 1], s=10, alpha=0.7)
        plt.title(f"Pareto Front - {name}")
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.grid()
        plt.savefig(f"{name}_pareto_front.png")
        plt.close()


if __name__ == "__main__":
    run_and_plot()
