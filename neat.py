import parameters
import math
import random
import numpy as np
from collections import deque

def sigmoid(x):
    """Modified sigmoid for better fine-tuning. """
    return 1/(1 + math.exp(-4.9*x))


class Genome:
    gene_innovation = 0  # Gene innovation number
    node_innovation = 0
    # Keeps track of all connections. Maps (in_node, out_node)
    # to the innovation number of the connection.
    gene_innovation_lookup = {}
    # Keeps track of nodes added through the add node mutation.
    # If Node 3 is added between Node 1 and Node 2, then the
    # dictionary maps (1, 2) to 3.
    node_innovation_lookup = {}

    # Create basic genome given list of inputs and list of outputs
    def __init__(self, nodes=None, genes=None):
        self.fitness = 0
        if nodes and genes:
            self.nodes = nodes
            self.genes = genes
            return

        self.nodes = {}  # Maps inn. number to gene object
        self.genes = {}  # Maps inn. number to node object

        for i in range(parameters.num_inputs + 1):  # + 1 because of bias term
            if i < Genome.node_innovation:  # If this isn't the first genome being created
                self.nodes[i] = Node(i)
            else:
                self.nodes[Genome.node_innovation] = Node(Genome.node_innovation)
                Genome.node_innovation += 1
        for j in range(parameters.num_inputs + 1, parameters.num_inputs + 1 + parameters.num_outputs):
            if j < Genome.node_innovation:  # If this isn't hte first genome being created
                o = j
            else:
                o = Genome.node_innovation
                Genome.node_innovation += 1
            self.nodes[o] = Node(o)
            for i in range(parameters.num_inputs + 1):
                weight = np.random.normal(0, parameters.init_weight_std)
                self.insert_gene(i, o, weight)

    def mutate_add_node(self):
        # Choose an enabled gene to split.
        while True:
            gene_numbers = list(self.genes.keys())
            gene_mutate = self.genes[random.choice(gene_numbers)]
            if gene_mutate.enable:
                break

        old_connection = (gene_mutate.in_node, gene_mutate.out_node)
        new_node = self.insert_node(old_connection[0], old_connection[1])
        self.insert_gene(old_connection[0], new_node.number, 1)
        self.insert_gene(new_node.number, old_connection[1], gene_mutate.weight)
        gene_mutate.enable = False

    def insert_gene(self, in_node, out_node, weight, enable=True):
        """Creates a gene, assigns it the correct innovation label, and adds it to the lookup table
        if necessary. """
        self.nodes[in_node].out_nodes.append(out_node)
        self.nodes[out_node].in_nodes.append(in_node)
        connection = (in_node, out_node)
        if connection in self.gene_innovation_lookup:  # If this gene had been created before
            innovation = self.gene_innovation_lookup[connection]
        else:
            # If this hasn't been created before, then add this creation to the lookup table
            innovation = Genome.gene_innovation
            self.gene_innovation_lookup[connection] = innovation
            Genome.gene_innovation += 1
        gene = Gene(in_node, out_node, weight, innovation, enable)
        self.genes[innovation] = gene
        self.nodes[out_node].incoming_genes.append(gene)
        return gene

    def insert_node(self, in_node, out_node, out_value=0):
        """Creates a node, assigns it the correct innovation label, and adds it to the lookup table.
        A new node is only added by 'splitting' a gene. (in_node, out_node) is the same in_node and
        out_node of the gene being 'split.' """
        key = (in_node, out_node)
        if key in self.node_innovation_lookup:
            innovation = self.node_innovation_lookup[key]
        else:
            innovation = Genome.node_innovation
            self.node_innovation_lookup[key] = innovation
            Genome.node_innovation += 1
        node = Node(innovation, out_value)
        self.nodes[innovation] = node
        return node

    def mutate_add_connection(self):
        """Adds a random connection in the neural network such that the result is still
        a feedforward neural network. """
        nodes = list(self.nodes.keys())

        for _ in range(min(len(self.nodes), 5)):  # Attempt this several times
            n1 = random.choice(nodes)
            n2 = random.choice(nodes)

            if n1 == n2:
                continue

            # If the proposed connection already exists
            if (n1, n2) in Genome.gene_innovation_lookup:
                innovation = Genome.gene_innovation_lookup[(n1, n2)]
                if innovation in self.genes:
                    continue

            # If the proposed connection doesn't point to an input node and doesn't create a cycle
            x = self.will_create_cycle(n1, n2)
            if n2 >= parameters.num_inputs and not self.will_create_cycle(n1, n2):
                weight = np.random.normal(0, parameters.init_weight_std)
                self.insert_gene(n1, n2, weight)
                return

    def will_create_cycle(self, a, b):
        """Tests whether adding the new link (a, b) will create a cycle in the network.
        Assumes that the graph currently contains no cycles (DAG). This can be done
        by running BFS to check if a path from B to A already exists in the current graph. """
        if a == b:
            return True
        goal = a
        closed = set()
        fringe = deque()
        fringe.append(b)
        while fringe:  # While fringe is not empty:
            current = fringe.popleft()
            if current == goal:
                return True
            if current not in closed:
                closed.add(current)
                for next_node in self.nodes[current].out_nodes:
                    fringe.append(next_node)
        return False

    def mutate_all_weights(self):
        for g in self.genes:
            if math.random() < parameters.p_perturb:  # Perturb the weight
                self.genes[g].weight += (math.random()*2 - 1) * parameters.max_perturb
            else:
                self.genes[g].weight = np.random.normal(0, parameters.init_weight_std)

    def print_debug_info(self):
        print("===========INFO=============")
        # print("gene lookup: ", self.gene_innovation_lookup)
        # print("node lookup: ", self.node_innovation_lookup)
        print("genes: ", self.genes)
        print("nodes: ", self.nodes)


class Node:
    def __init__(self, innovation, out_value=0):
        self.number = innovation
        self.out_nodes = []
        self.in_nodes = []
        self.incoming_genes = []
        self.value = out_value

    def __str__(self):
        # return "||n:" + str(self.number) + ", i:" + str(self.in_nodes) + ", o:" + str(self.out_nodes) + "|| "
        return "||n:" + str(self.number) + ", v:" + str(self.value) + "|| "

    __repr__ = __str__


class Gene:
    def __init__(self, in_node, out_node, weight, innovation, enable=True):
        self.number = innovation
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enable = enable

    def copy(self):
        return Gene(self.in_node, self.out_node, self.weight, self.number, self.enable)

    def __str__(self):
        return "||n:" + str(self.number) + ", i: " + str(self.in_node) + ", o:" \
               + str(self.out_node) + ", w:" + str(self.weight) + ", e:" + str(self.enable) + "|| "

    __repr__ = __str__


class NeuralNetwork:
    """Converts a genome to a neural network, which takes in inputs
    to return the output. """
    def __init__(self, genome):
        self.nodes = genome.nodes
        # Topologically sorts the nodes in O(|V| + |E|) time.

        self.ordered_nodes = []
        in_degree_list = {}

        for n in genome.nodes:
            in_degree = len(genome.nodes[n].in_nodes)
            in_degree_list[n] = in_degree

        # Currently the input nodes are sources
        current_source_nodes = range(parameters.num_inputs + 1)
        next_source_nodes = []

        while current_source_nodes:
            self.ordered_nodes += current_source_nodes
            for u in current_source_nodes:
                for v in genome.nodes[u].out_nodes:
                    in_degree_list[v] -= 1
                    if in_degree_list[v] == 0:
                        next_source_nodes.append(v)
            current_source_nodes = next_source_nodes
            next_source_nodes = []

    def evaluate(self, input):
        if len(input) != parameters.num_inputs:
            print("ERROR: Invalid input size")
        self.nodes[0].value = 1  # Bias term
        for i in range(parameters.num_inputs):
            self.nodes[i + 1].value = input[i]
        for j in range(parameters.num_inputs + 1, len(self.ordered_nodes)):
            node = self.ordered_nodes[j]
            total = 0
            for gene in self.nodes[node].incoming_genes:
                if gene.enable:
                    total += gene.weight * self.nodes[gene.in_node].value
            self.nodes[node].value = sigmoid(total)

        outputs = []
        for o in range(parameters.num_inputs + 1, parameters.num_inputs + 1 + parameters.num_outputs):
            outputs.append(self.nodes[o].value)
        return outputs


def crossover(genome1, genome2):
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
            gene = random.choice([g1, g2]).copy()
            if not g1.enable and not g2.enable and random.random() < parameters.p_enable_if_both_parents_disabled:
                gene.enable = True

        composite_genes[i] = gene
        # Generate nodes from gene
        if gene.in_node not in composite_nodes:
            composite_nodes[gene.in_node] = Node(gene.in_node)
        composite_nodes[gene.in_node].out_nodes.append(gene.out_node)
        if gene.out_node not in composite_nodes:
            composite_nodes[gene.out_node] = Node(gene.out_node)
        composite_nodes[gene.out_node].in_nodes.append(gene.in_node)

    return Genome(composite_nodes, composite_genes)


def get_dict_value(val, dict):
    try:
        return dict[val]
    except KeyError:
        return None

class Species:
    def __init__(self):
        self.representative = None
        self.genomes = []


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
    return parameters.c1*excess/N + parameters.c2*disjoint/N + parameters.c3*sum_of_differences/N

def find_species(species, genome):
    """Returns the species that the genome belongs to. If
    there aren't any species that it belongs to, it returns None. """
    for s in species:
        if delta(s.representative, genome) < parameters.delta_threshold:
            return s
    return None

# g1 = Genome()
# g2 = Genome()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.mutate_add_node()
# g1.print_debug_info()
# print(delta(g1, g2))