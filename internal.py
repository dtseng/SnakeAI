import parameters
import math
import numpy as np
from collections import deque
import pickle

def sigmoid(x):
    return 1/(1 + math.exp(-1*x))


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

        for i in range(parameters.num_inputs):
            if i < Genome.node_innovation:  # If this isn't the first genome being created
                self.nodes[i] = Node(i, np.random.normal(0, parameters.bias_init_std))
            else:
                self.nodes[Genome.node_innovation] = Node(Genome.node_innovation, np.random.normal(0, parameters.bias_init_std))
                Genome.node_innovation += 1
        for j in range(parameters.num_inputs, parameters.num_inputs + parameters.num_outputs):
            if j < Genome.node_innovation:  # If this isn't the first genome being created
                o = j
            else:
                o = Genome.node_innovation
                Genome.node_innovation += 1
            self.nodes[o] = Node(o, np.random.normal(0, parameters.bias_init_std))
            for i in range(parameters.num_inputs):
                # weight = np.random.normal(0, parameters.init_weight_std)
                weight = np.random.rand()*4 - 2
                self.insert_gene(i, o, weight)

    def mutate_add_node(self):
        # Choose an enabled gene to split.
        for attempt in range(5*len(self.genes)):  # Number of attempts to find a valid gene
            gene_numbers = list(self.genes.keys())
            gene_mutate = self.genes[np.random.choice(gene_numbers)]
            if gene_mutate.enable:
                old_connection = (gene_mutate.in_node, gene_mutate.out_node)
                break
        if attempt == 5*len(self.genes) - 1:
            print("Cannot add new node")  # Not an error, just for info purposes.
            return
        # print("old connection: " + str(old_connection))
        # print("nodes: " + str(self.nodes))
        # print("genes; " + str(self.genes))
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
        if connection in Genome.gene_innovation_lookup:  # If this gene had been created before
            innovation = Genome.gene_innovation_lookup[connection]
            if innovation in self.genes:
                print("attempt in: " + str(in_node) + " attempt out: " + str(out_node))
                with open("anomalies/what", 'wb') as output:
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
                raise ValueError('Attempted to insert existent gene. ')
        else:
            # If this hasn't been created before, then add this creation to the lookup table
            innovation = Genome.gene_innovation
            Genome.gene_innovation_lookup[connection] = innovation
            Genome.gene_innovation += 1
        gene = Gene(in_node, out_node, weight, innovation, enable)
        self.genes[innovation] = gene
        self.nodes[out_node].incoming_genes.append(gene)
        return gene

    def insert_node(self, in_node, out_node):
        """Creates a node, assigns it the correct innovation label, and adds it to the lookup table.
        A new node is only added by 'splitting' a gene. (in_node, out_node) is the same in_node and
        out_node of the gene being 'split.' """
        key = (in_node, out_node)
        if key in Genome.node_innovation_lookup:
            innovation = Genome.node_innovation_lookup[key]
        else:
            innovation = Genome.node_innovation
            Genome.node_innovation_lookup[key] = innovation
            Genome.node_innovation += 1
        node = Node(innovation, np.random.normal(0, parameters.bias_init_std))
        self.nodes[innovation] = node
        return node

    def mutate_add_connection(self):
        """Adds a random connection in the neural network such that the result is still
        a feedforward neural network. """
        nodes = list(self.nodes.keys())

        for _ in range(min(10*len(self.nodes), 5)):  # Attempt this several times
            n1 = np.random.choice(nodes)
            n2 = np.random.choice(nodes)

            if n1 == n2:
                continue

            # If the proposed connection already exists
            if (n1, n2) in Genome.gene_innovation_lookup and Genome.gene_innovation_lookup[(n1, n2)] in self.genes:
                continue

            # If the proposed connection doesn't point to an input node and doesn't create a cycle
            if n2 >= parameters.num_inputs and not self.will_create_cycle(n1, n2):
                weight = np.random.normal(0, parameters.weight_init_std)
                # weight = np.random.rand()*4 - 2

                self.insert_gene(n1, n2, weight)
                return

    def mutate_delete_connection(self):
        if len(self.genes) == 1:
            return
        gene = np.random.choice(list(self.genes.keys()))
        in_node = self.genes[gene].in_node
        out_node = self.genes[gene].out_node
        self.nodes[in_node].out_nodes.remove(out_node)
        self.nodes[out_node].in_nodes.remove(in_node)
        self.nodes[out_node].incoming_genes.remove(self.genes[gene])
        del self.genes[gene]
        assert(len(self.genes) > 0)


    def mutate_delete_node(self):
        del_node = np.random.choice(list(self.nodes.keys()))
        if del_node < parameters.num_inputs + parameters.num_outputs:  # Can't delete an input or output
            return
        if len(self.nodes[del_node].in_nodes) + len(self.nodes[del_node].out_nodes) == len(self.genes):
            return

        genes_to_delete = set()
        for g in self.genes:
            if del_node == self.genes[g].in_node or del_node == self.genes[g].out_node:
                genes_to_delete.add(g)
        for out in self.nodes[del_node].out_nodes:
            self.nodes[out].in_nodes.remove(del_node)
        for in_n in self.nodes[del_node].in_nodes:
            self.nodes[in_n].out_nodes.remove(del_node)

        for key in genes_to_delete:
            o_node = self.genes[key].out_node
            self.nodes[o_node].incoming_genes.remove(self.genes[key])
            del self.genes[key]

        del self.nodes[del_node]

        # in_node_sum = 0
        # out_node_sum = 0
        # for i in self.nodes:
        #     in_node_sum += len(self.nodes[i].in_nodes)
        #     out_node_sum += len(self.nodes[i].out_nodes)
        #
        # assert in_node_sum == len(self.genes)
        #
        #
        # assert in_node_sum == out_node_sum


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
        # i, o = a, b
        # if i == o:
        #     return True
        #
        # visited = {o}
        # while True:
        #     num_added = 0
        #     # for a, b in connections:
        #     for g in self.genes:
        #         a, b = self.genes[g].in_node, self.genes[g].out_node
        #         if a in visited and b not in visited:
        #             if b == i:
        #                 return True
        #
        #             visited.add(b)
        #             num_added += 1
        #
        #     if num_added == 0:
        #         return False

    def mutate_all_weights(self):
        for g in self.genes:
            if np.random.rand() < parameters.p_perturb:  # Perturb the weight
                self.genes[g].weight += (np.random.rand()*2 - 1) * parameters.max_perturb
            else:
                self.genes[g].weight = np.random.normal(0, parameters.weight_init_std)
                # self.genes[g].weight = np.random.rand()*4 - 2

    def mutate(self):
        """General function for mutation. Uses all the other mutation functions. """
        if np.random.rand() < parameters.new_node_rate:
            self.mutate_add_node()

        if np.random.rand() < parameters.del_node_rate:
            self.mutate_delete_node()

        if np.random.rand() < parameters.new_link_rate:
            self.mutate_add_connection()

        if np.random.rand() < parameters.del_link_rate:
            self.mutate_delete_connection()

        for g in self.genes:
            self.genes[g].mutate_weight()
        for n in self.nodes:
            self.nodes[n].mutate_bias()
        for k in self.genes:
            if k != self.genes[k].number:
                raise ValueError('Mismatched genome number with dictionary. ')

    def print_debug_info(self):
        print("===========INFO=============")
        # print("gene lookup: ", Genome.gene_innovation_lookup)
        # print("node lookup: ", Genome.node_innovation_lookup)
        print("genes: ", self.genes)
        print("nodes: ", self.nodes)


class Node:
    def __init__(self, innovation, bias, out_value=0):
        self.number = innovation
        self.out_nodes = []
        self.in_nodes = []
        self.incoming_genes = []
        self.value = out_value
        self.bias = bias

    def copy(self):
        new_node = Node(self.number, self.bias)
        new_node.out_nodes = list(self.out_nodes)
        new_node.in_nodes = list(self.in_nodes)
        return new_node

    def mutate_bias(self):
        r = np.random.rand()
        if r < parameters.bias_replace_rate:
            self.bias = np.random.normal(0, parameters.bias_init_std)
        if r < parameters.bias_replace_rate + parameters.bias_mutate_rate:
            self.bias = self.bias + np.random.normal(0.0, parameters.bias_mutate_std)

    def __str__(self):
        # return "||n:" + str(self.number) + ", i:" + str(self.in_nodes) + ", o:" + str(self.out_nodes) + "|| "
        return "||n:" + str(self.number) + ", in: " + str(self.in_nodes) + ", out: " + str(self.out_nodes) + ", bias: " + str(self.bias)

    __repr__ = __str__


class Gene:
    def __init__(self, in_node, out_node, weight, innovation, enable=True):
        self.number = innovation
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enable = enable

    def mutate_weight(self):
        r = np.random.rand()
        if r < parameters.weight_replace_rate:
            self.weight = np.random.normal(0, parameters.weight_init_std)
        if r < parameters.weight_replace_rate + parameters.weight_mutate_rate:
            self.weight = self.weight + np.random.normal(0.0, parameters.weight_mutate_std)

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
        self.genome = genome
        # seen_so_far = set()  # For O(1) access
        # not_seen_yet = set(genome.nodes)
        # self.genome = genome
        # self.ordered_nodes = []  # Topologically sorted.
        #
        # # A bit slow. Think of faster way later.
        # for _ in range(len(genome.nodes)):
        #     remove_set = set()
        #     for node in not_seen_yet:
        #         if all(u in seen_so_far for u in genome.nodes[node].in_nodes):
        #             seen_so_far.add(node)
        #             # not_seen_yet.remove(node)
        #             remove_set.add(node)
        #             self.ordered_nodes.append(node)
        #     not_seen_yet = not_seen_yet.difference(remove_set)
        # if not_seen_yet:
        #     print("baddddddddddd " + str(len(not_seen_yet)))
        # if len(self.nodes) != len(self.ordered_nodes):
        #     print("bad bad bad bad bad badbf adb adb adb ")


        # Topologically sorts the nodes in O(|V| + |E|) time.

        self.ordered_nodes = []
        in_degree_list = {}

        current_source_nodes = []

        for n in genome.nodes:
            genome.nodes[n].value = 0
            in_degree = len(genome.nodes[n].in_nodes)
            if in_degree == 0:
                current_source_nodes.append(n)
            in_degree_list[n] = in_degree

        # assert all(i in current_source_nodes for i in range(parameters.num_inputs))

        # Currently the input nodes are sources
        # current_source_nodes = range(parameters.num_inputs)
        next_source_nodes = []
        finished = set()
        while current_source_nodes:
            self.ordered_nodes += current_source_nodes
            for u in current_source_nodes:
                for v in genome.nodes[u].out_nodes:
                    in_degree_list[v] -= 1
                    if in_degree_list[v] == 0:
                        next_source_nodes.append(v)
            current_source_nodes = next_source_nodes
            next_source_nodes = []
        # for i in in_degree_list:
        #     if in_degree_list[i] != 0:
        #         print("badddd" + str(in_degree_list[i]))

        if len(self.nodes) != len(self.ordered_nodes):
            with open("anomalies/bad_genome", 'wb') as output:
                pickle.dump(self.genome, output, pickle.HIGHEST_PROTOCOL)
            print(genome.genes)
            for n in self.nodes:
                print(str(n) + ": " + "in: " + str(self.nodes[n].in_nodes) + " out: " + str(self.nodes[n].out_nodes))
            raise ValueError('Topological sort failed. ')
        assert (all(len(self.genome.nodes[i].incoming_genes) == len(self.genome.nodes[i].in_nodes) for i in self.genome.nodes))

    def evaluate(self, input):
        if len(input) != parameters.num_inputs:
            raise ValueError('Incorrect input size.')
        for i in range(parameters.num_inputs):
            self.nodes[i].value = input[i]
        for j in range(parameters.num_inputs, len(self.ordered_nodes)):
            node = self.ordered_nodes[j]
            total = 0
            for gene in self.nodes[node].incoming_genes:
                if gene.enable:
                    total += gene.weight * self.nodes[gene.in_node].value
            self.nodes[node].value = sigmoid(total + self.nodes[node].bias)

        outputs = []
        for o in range(parameters.num_inputs, parameters.num_inputs + parameters.num_outputs):
            outputs.append(self.nodes[o].value)

        # for _ in range(2*len(self.ordered_nodes)):
        #     for j in range(parameters.num_inputs, len(self.ordered_nodes)):
        #         node = self.ordered_nodes[j]
        #         total = 0
        #         for gene in self.nodes[node].incoming_genes:
        #             if gene.enable:
        #                 total += gene.weight * self.nodes[gene.in_node].value
        #         self.nodes[node].value = sigmoid(total)
        #
        # outputs2 = []
        # for j in range(parameters.num_inputs, parameters.num_inputs + parameters.num_outputs):
        #     outputs2.append(self.nodes[j].value)
        #
        # if not all(outputs2[i] == outputs[i] for i in range(len(outputs))):
        #     print(outputs2)
        #     print(outputs)
        #     print(self.nodes)
        #     with open("anomalies/weird_genome", 'wb') as output:
        #         pickle.dump(self.genome, output, pickle.HIGHEST_PROTOCOL)
        #
        #     raise ValueError("Bad evaluation of network.")
        return outputs


class Species:
    def __init__(self, representative=None):
        self.representative = representative
        self.genomes = []
        self.best_fitness = float("-inf")
        self.stagnate_generations = 0
        self.sum_adj_fitness = 0
        self.spawn_amount = 0
        self.best_genome = None

    def __str__(self):
        return "(" + str(len(self.representative.nodes)) + ", " + str(len(self.representative.genes)) + ") "


    __repr__ = __str__
