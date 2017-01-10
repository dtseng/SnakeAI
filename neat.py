import parameters
import math
import random
import numpy as np

def sigmoid(x):
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
    def __init__(self, inputs):  # TODO: Fix this, using the new methods you just created
        if len(inputs) != parameters.num_inputs:
            print("ERROR: Invalid number of inputs")
        inputs = [1] + inputs  # Bias term
        self.nodes = {}  # Maps inn. number to gene object
        self.genes = {}  # Maps inn. number to node object
        for i in inputs:
            self.nodes[Genome.node_innovation] = Node(Genome.node_innovation, i)
            Genome.node_innovation += 1
        for _ in range(parameters.num_outputs):
            o = Genome.node_innovation
            self.nodes[o] = Node(o)
            for i in range(len(inputs)):
                weight = np.random.normal(0, parameters.init_weight_std)
                self.insert_gene(i, o, weight)
            Genome.node_innovation += 1

    def mutate_add_node(self):
        gene_numbers = list(self.genes.keys())
        gene_mutate = self.genes[random.choice(gene_numbers)]
        old_connection = (gene_mutate.in_node, gene_mutate.out_node)
        new_node = self.insert_node(old_connection[0], old_connection[1])
        self.insert_gene(old_connection[0], new_node.innovation, 1)
        self.insert_gene(new_node.innovation, old_connection[1], gene_mutate.weight)
        gene_mutate.enable = False

    def insert_gene(self, in_node, out_node, weight, enable=True):
        """Creates a gene, assigns it the correct innovation label, and adds it to the lookup table
        if necessary. """
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
        return gene

    def insert_node(self, in_node, out_node, out_value=0):
        """Creates a node, assigns it the correct innovation label, and adds it to the lookup table.
        A new node is only added by splitting a gene. (in_node, out_node) is the same in_node and
        out_node of the gene being split. """
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
        pass

    def print_debug_info(self):
        print("gene lookup: ", self.gene_innovation_lookup)
        print("node lookup: ", self.node_innovation_lookup)
        print("genes: ", self.genes)
        print("nodes: ", self.nodes)


class Node:
    def __init__(self, innovation, out_value=0):
        self.innovation = innovation
        self.out_value = out_value

    def __str__(self):
        return "||n:" + str(self.innovation) + ", o:" + str(self.out_value) + "|| "

    __repr__ = __str__


class Gene:
    def __init__(self, in_node, out_node, weight, innovation, enable=True):
        self.number = innovation
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enable = enable

    def __str__(self):
        return "||n:" + str(self.number) + ", i: " + str(self.in_node) + ", o:" \
               + str(self.out_node) + ", w:" + str(self.weight) + ", e:" + str(self.enable) + "|| "

    __repr__ = __str__

inputs = [4]
g = Genome(inputs)
g.print_debug_info()
g.mutate_add_node()
print("===============")
g.print_debug_info()