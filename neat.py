class Genome:
    def __init__(self):
        self.connections = []
        self.neurons = []

    def mutate_add_node(self):
        pass

    def mutate_add_link(self):
        pass

class Neuron:
    def __init__(self):


class Link:
    def __init__(self, in_node, out_node, weight, innovation, enable=True):
        self.innovation = innovation
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enable = enable







