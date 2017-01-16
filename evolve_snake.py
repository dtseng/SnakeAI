import parameters
import internal
import pickle
from medoids import k_medoids
import numpy as np
import snake_game


def crossover(genome1, genome2):
    """Implements the crossover functionality of the NEAT algorithm."""
    if np.random.rand() < 0.5:  # Increase randomness
        genome1, genome2 = genome2, genome1

    if genome2.fitness > genome1.fitness:
        genome1, genome2, = genome2, genome1

    genes1 = genome1.genes
    genes2 = genome2.genes
    composite_genes = {}
    composite_nodes = {}

    assert genome1.fitness >= genome2.fitness

    # Composite genome has the same connections as the more fit genome
    for g in genes1:
        if g not in genes2:
            composite_genes[g] = genes1[g].copy()
        else:
            composite_genes[g] = np.random.choice([genes1[g], genes2[g]]).copy()
            if not genes1[g].enable and not genes2[g].enable and \
                            np.random.rand() < parameters.p_enable_if_both_parents_disabled:
                composite_genes[g].enable = True

    # Composite genome has the same nodes as the more fit genome
    for n in genome1.nodes:
        composite_nodes[n] = genome1.nodes[n].copy()
        composite_nodes[n].incoming_genes = []
        for incoming in genome1.nodes[n].incoming_genes:
            incoming_key = incoming.number
            composite_nodes[n].incoming_genes.append(composite_genes[incoming_key])
        if n in genome1.nodes and n in genome2.nodes:
            composite_nodes[n].bias = np.random.choice([genome1.nodes[n], genome2.nodes[n]]).bias

    return internal.Genome(composite_nodes, composite_genes)


def delta(genome1, genome2):
    """This is the delta function used to determine the similarity between two
    genomes. The higher the result, the more dissimilar the two genomes are. """
    if max(genome2.genes.keys()) > max(genome1.genes.keys()):
        genome1, genome2 = genome2, genome1
    assert (max(genome1.genes.keys()) >= max(genome2.genes.keys()))

    g1 = set(genome1.genes.keys())
    g2 = set(genome2.genes.keys())
    excess_cutoff = max(g2)
    excess = 0
    disjoint = 0
    sum_of_differences = 0
    N = min(len(g1), len(g2))

    for g in g1:
        if g in g2:
            sum_of_differences += abs(genome1.genes[g].weight - genome2.genes[g].weight)
        else:
            if g > excess_cutoff:
                excess += 1
            else:
                disjoint += 1
    for g in g2:
        if g not in g1:
            disjoint += 1

    return parameters.c1 * excess / N + parameters.c2 * disjoint / N + parameters.c3 * sum_of_differences / N

def find_species(population, genome):
    """population is a list of species. Returns the species that the genome belongs to. If
    there aren't any species that it belongs to, it creates a new species. """
    for s in population:
        if delta(s.representative, genome) < parameters.delta_threshold:
            # print(delta(s.representative, genome), end='')
            # print("(" + str(len(genome.nodes)) + ", " + str(len(genome.genes)) + ")")
            return s
    new_species = internal.Species(genome)
    population.append(new_species)
    return new_species

def run_genome_in_game(genome, display_graphics=False):
    """Evaluates the fitness of a genome by using it in the snake game. """
    nn = internal.NeuralNetwork(genome)
    game = snake_game.Game(nn.evaluate, display_graphics)
    game_time = 0
    while game_time < 10000 and game.get_continue_status():  # Run the game
        game.step()
        genome.fitness = 15 * game.snake.fitness + 0.5 * (game.manhattan_distance_to_food())
        # genome.fitness = game_time
        game_time += 1
    return genome.fitness

def evaluate_population(population):
    """Finds the fitness for all genomes in the population. The population is given as
    a list of species."""
    max_fitness = float("-inf")
    average_fitness = 0
    best_genome = None
    for species in population:
        for genome in species.genomes:
            current = run_genome_in_game(genome)
            average_fitness += current
            if current > max_fitness:
                max_fitness = current
                best_genome = genome
    average_fitness /= parameters.population_size
    return max_fitness, average_fitness, best_genome

def init_population():
    """Returns a one-element list of one species. This species contains the entire population
    for the first generation. """
    all_genomes = []
    for _ in range(parameters.population_size):
        genome = internal.Genome()
        all_genomes.append(genome)
    species = internal.Species(np.random.choice(all_genomes))
    species.genomes = all_genomes
    return [species]

def update_spawn_amounts(population):
    """Finding spawn amounts. Each species' spawn amount is proportional to the sum of its
    genomes' adjusted fitness."""
    stagnated = set()
    total_sum_adj_fitness = 0
    for species in population:
        N = len(species.genomes)
        for genome in species.genomes:
            species.sum_adj_fitness += genome.fitness / N

        species.best_genome = max(species.genomes, key=lambda x: x.fitness)
        max_fitness = species.best_genome.fitness
        if max_fitness <= species.best_fitness:  # Species didn't improve
            species.stagnate_generations += 1
        else:
            species.stagnate_generations = 0  # Restart stagnate count
            species.best_fitness = max_fitness
        if species.stagnate_generations >= parameters.stagnate_threshold:  # Species has stagnated
            stagnated.add(species)
            print("STAGNATED: Removed 1 species.")
        else:
            total_sum_adj_fitness += species.sum_adj_fitness

    for species in stagnated:
        population.remove(species)
    for species in population:
        species.spawn_amount = max(3, int(
            parameters.population_size * species.sum_adj_fitness / total_sum_adj_fitness))
    return population

def next_generation_species(species):
    """Returns the next generation (list) of genomes given the species. The size of the next generation
    will be species.spawn_amount"""
    spawn_amount = species.spawn_amount
    sorted_genomes = sorted(species.genomes, key=lambda g: g.fitness, reverse=True)
    probability_chosen = np.array([g.fitness for g in species.genomes])
    probability_chosen = probability_chosen / sum(probability_chosen)  # Normalize

    N = len(species.genomes)
    new_generation = sorted_genomes[0:max(2, int(parameters.keep_best_amount * N))]  # Keep top fraction of species
    while len(new_generation) < spawn_amount:
        g1 = np.random.choice(species.genomes, p=probability_chosen)
        g2 = np.random.choice(species.genomes, p=probability_chosen)
        new_generation.append(crossover(g1, g2))

    for i in range(2, len(new_generation)):  # The top 2 genomes do not get mutated
        new_generation[i].mutate()
    return new_generation

def next_generation_population(population):
    """Returns the next generation's list of species given this generation's list of species. """
    next_genomes = []  # Contains a list of all genomes
    for species in population:
        next_genomes += next_generation_species(species)
        species.genomes = []

    # Assign each genome to a species
    for genome in next_genomes:
        best_fit = find_species(population, genome)
        best_fit.genomes.append(genome)
    next_population = []
    for species in population:
        if species.genomes:  # If the species still has something assigned to it.
            next_population.append(species)
    population = next_population

    # Update representatives for each genome
    for species in population:
        species.representative = np.random.choice(species.genomes)
    return population

def evolution():
    """Puts everything together to evolve the snake AI. """
    population = init_population()
    best_fitness_overall = float("-inf")
    for gen_number in range(parameters.num_generations):
        total_size_population = 0
        print("=================Generation " + str(gen_number) + "===================")
        best_fitness_population, average_fitness, best_genome = evaluate_population(population)
        if best_fitness_population > best_fitness_overall:
            best_fitness_overall = best_fitness_population
            with open("genomes/generation " + str(gen_number) + " fitness " + str(best_genome.fitness),
                      'wb') as output:
                pickle.dump(best_genome, output, pickle.HIGHEST_PROTOCOL)
        print("Number of species: " + str(len(population)))
        print("Sizes: ", end='')
        for species in population:
            print(len(species.genomes), " ", end='')
            total_size_population += len(species.genomes)
        print()
        print("Total size: " + str(total_size_population))
        print("Best fitness: " + str(best_fitness_population))
        print("Average fitness: " + str(average_fitness))
        # print()
        internal.Genome.node_innovation_lookup = {}
        # internal.Genome.gene_innovation_lookup = {}
        population = update_spawn_amounts(population)
        population = next_generation_population(population)


def test_genome_in_game(filename):
    """Loads the genome from file, and displays the snake AI as played by the
    genome. """
    with open(filename, 'rb') as file:
        genome = pickle.load(file)
    fitness = run_genome_in_game(genome, display_graphics=True)
    print("fitness: " + str(fitness))

np.random.seed(20)  # For debugging purposes
evolution()
# test_genome_in_game("samples2/generation 203 fitness 376.0")
# show_genome_capabilities("genomes/generation 78 fitness 92.5")