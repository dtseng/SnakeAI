import parameters
import internal
import pickle
from medoids import k_medoids
import numpy as np
import silent_game

def crossover(genome1, genome2):
    """Implements the crossover functionality of the NEAT algorithm."""
    if np.random.rand() < 0.5:
        temp = genome1
        genome1 = genome2
        genome2 = temp

    genes1 = genome1.genes
    genes2 = genome2.genes
    better = max([genome1, genome2], key=lambda x: x.fitness)
    composite_genes = {}
    composite_nodes = {}

    largest_innovation_number = max(list(genes1.keys()) + list(genes2.keys()))
    for i in range(largest_innovation_number + 1):
        g1 = get_dict_value(i, genes1)
        g2 = get_dict_value(i, genes2)

        if not (g1 or g2):
            continue
        elif g1 and not g2 and better == genome1:
            gene = g1.copy()
        elif g2 and not g1 and better == genome2:
            gene = g2.copy()
        elif g1 and g2:
            gene = np.random.choice([g1, g2]).copy()
            if not g1.enable and not g2.enable and np.random.rand() < parameters.p_enable_if_both_parents_disabled:
                gene.enable = True
        else:
            continue
        composite_genes[i] = gene
    for n in better.nodes:
        composite_nodes[n] = better.nodes[n].copy()
        for incoming in better.nodes[n].incoming_genes:
            incoming_key = incoming.number
            composite_nodes[n].incoming_genes.append(composite_genes[incoming_key])
        if n in genome1.nodes and n in genome2.nodes:
            composite_nodes[n].bias = np.random.choice([genome1.nodes[n], genome2.nodes[n]]).bias
    # assert(all(len(composite_nodes[i].incoming_genes) == len(composite_nodes[i].in_nodes) for i in composite_nodes))


    # print("--")
    # print("genome1: " + str(genome1.nodes) + "fitness: " + str(genome1.fitness))
    # print("genome2: " + str(genome2.nodes) + "fitness: " + str(genome2.fitness))
    # print("genome3: " + str(composite_nodes))

    for g in composite_genes:
        if g != composite_genes[g].number:
            print("ERROR -1")
            with open("anomalies/bad_genome1", 'wb') as output:
                pickle.dump(genome1, output, pickle.HIGHEST_PROTOCOL)
            with open("anomalies/bad_genome2", 'wb') as output:
                pickle.dump(genome2, output, pickle.HIGHEST_PROTOCOL)
            with open("anomalies/bad_composite", 'wb') as output:
                pickle.dump(composite_genes, output, pickle.HIGHEST_PROTOCOL)
            raise ValueError('Mismatched genome number with dictionary. ')
    return internal.Genome(composite_nodes, composite_genes)


def get_dict_value(val, dict):
    try:
        return dict[val]
    except KeyError:
        return None


def delta(genome1, genome2):
    g1 = set(genome1.genes.keys())
    g2 = set(genome2.genes.keys())
    max1 = max(g1)
    max2 = max(g2)

    excess = 0
    disjoint = 0
    sum_of_differences = 0
    N = min(len(g1), len(g2))
    # if N < 20:
        # N = 1

    for current in g1.union(g2):
        if current in g1 and current in g2:
            sum_of_differences += abs(genome1.genes[current].weight - genome2.genes[current].weight)
        elif current in g1 and current not in g2:
            if current > max2:
                excess += 1
            else:
                disjoint += 1
        elif current in g2 and current not in g1:
            if current > max1:
                excess += 1
            else:
                disjoint += 1
        else:
            raise ValueError("Bad condition in detla function.")
    return parameters.c1*excess/N + parameters.c2*disjoint/N + parameters.c3*sum_of_differences/N


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


def get_genome_fitness(genome):
    nn = internal.NeuralNetwork(genome)
    game = silent_game.Game(nn.evaluate)
    game_time = 0
    while game_time < 10000 and game.get_continue_status():  # Run the game
        game.step()
        genome.fitness = 15 * game.snake.fitness + 0.5 * (game.manhattan_distance_to_food())
        # genome.fitness = game_time
        game_time += 1
    return genome.fitness


def evaluate_population(population):
    max_fitness = float("-inf")
    average_fitness = 0
    for species in population:
        for genome in species.genomes:
            current = get_genome_fitness(genome)
            average_fitness += current
            max_fitness = max(max_fitness, current)
    average_fitness /= parameters.population_size
    return max_fitness, average_fitness


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
            species.sum_adj_fitness += genome.fitness/N

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
        species.spawn_amount = max(3, int(parameters.population_size * species.sum_adj_fitness/total_sum_adj_fitness))
    return population


def next_generation_species(species):
    """Returns the next generation (list) of genomes given the species. The size of the next generation
    will be species.spawn_amount"""
    spawn_amount = species.spawn_amount
    sorted_genomes = sorted(species.genomes, key=lambda g: g.fitness, reverse=True)
    probability_chosen = np.array([g.fitness for g in species.genomes])
    probability_chosen = probability_chosen / sum(probability_chosen)  # Normalize

    N = len(species.genomes)
    new_generation = sorted_genomes[0:max(2, int(parameters.keep_best_amount*N))]  # Keep top fraction of species
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
    """
    #============= Medoids
    diameter, med = k_medoids(next_genomes, k=5, distance=delta, spawn=1, verbose=False)
    next_population = []
    for m in med:
        sp = internal.Species()
        sp.genomes = m.elements
        next_population.append(sp)

    population = next_population"""
    #============
    # Update representatives for each genome
    for species in population:
        species.representative = np.random.choice(species.genomes)
    return population


def evolution():
    """Puts everything together to evolve the snake AI. """
    population = init_population()
    for gen_number in range(parameters.num_generations):
        print("=================Generation " + str(gen_number) + "===================")
        best_fitness, average_fitness = evaluate_population(population)
        print("Number of species: " + str(len(population)))
        for species in population:
            s = sorted(species.genomes, key=lambda x: x.fitness, reverse=True)
            # best = s[0]
            # bf = best.fitness
            # for genome in s:
            #     print("(" + str(len(genome.nodes)) + ", " + str(len(genome.genes)) + ")" + ": " + str(genome.fitness) + " ")
            # print(species, end='')
        # print()
        print("Sizes: ", end='')
        for species in population:
            print(len(species.genomes), " ", end='')
        print()
        print("Best fitness: " + str(best_fitness))
        print("Average fitness: " + str(average_fitness))
        # print()
        internal.Genome.node_innovation_lookup = {}
        # internal.Genome.gene_innovation_lookup = {}
        population = update_spawn_amounts(population)
        population = next_generation_population(population)

np.random.seed(20)
evolution()
