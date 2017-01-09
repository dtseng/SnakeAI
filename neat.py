import parameters
import math
import numpy as np

def sigmoid(x):
    return 1/(1 + math.exp(-4.9*x))


class Genome:
    gene_innovation = 0 # Gene innovation number
    node_innovation = 0
    # Keeps track of all connections. Maps (in_node, out_node)
    # to the innovation number of the connection.
    gene_innovation_lookup = {}
    # Keeps track of nodes added through the add node mutation.
    # If Node 3 is added between Node 1 and Node 2, then the
    # dictionary maps (1, 2) to 3.
    node_innovation_lookup = {}

    # Create basic genome given list of inputs and list of outputs
    def __init__(self, inputs):
        if len(inputs) != parameters.num_inputs:
            print("ERROR: Invalid number of inputs")
        inputs = [1] + inputs  # Bias term
        self.nodes = {}  # Maps inn. number to gene object
        self.connections = {}  # Maps inn. number to node object
        for i in inputs:
            self.nodes[Genome.node_innovation] = Node(Genome.node_innovation, i)
            Genome.node_innovation += 1
        for o in range(parameters.num_outputs):
            n = Genome.node_innovation
            self.nodes[n] = Node(n)
            for i in range(len(inputs)):
                weight = np.random.normal(0, parameters.init_weight_std)
                self.connections[Genome.gene_innovation] = Gene(i, n, weight, Genome.gene_innovation)
                Genome.gene_innovation_lookup[(i, o)] = Genome.gene_innovation
                Genome.gene_innovation += 1
            Genome.node_innovation += 1

    def mutate_add_node(self):
        pass

    def mutate_add_link(self):
        pass


class Node:
    def __init__(self, innovation, out_value=0):
        self.number = innovation
        self.out_value = out_value

    def __str__(self):
        return "n: " + str(self.number) + ", o:" + str(self.out_value)

    __repr__ = __str__

class Gene:
    def __init__(self, in_node, out_node, weight, innovation, enable=True):
        self.innovation = innovation
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enable = enable

    def __str__(self):
        return "i: " + str(self.in_node) + ", o:" + str(self.out_node) + ", w:" + str(self.weight) \
               + ", n:" + str(self.innovation)

    __repr__ = __str__

inputs = [4, 5, 6]
g = Genome(inputs)