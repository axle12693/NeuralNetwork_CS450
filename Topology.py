import math


class Topology:
    def __init__(self):
        self.layers = []
        self.implemented = {"Sigmoid": ("Sigmoid",)}  # key = layer j, value = layer k
        self.implementation = {"Sigmoid": Topology.sigmoid}

    def add_layer(self, num_neurons, activation_type="Sigmoid"):
        if activation_type not in self.implemented[self.layers[len(self.layers) - 1]]:
            raise Exception("Functionality not built yet for transition from " + self.layers[len(self.layers) - 1] + " to " + activation_type)
        self.layers.append((self.implementation[activation_type], num_neurons))

    @staticmethod
    def sigmoid(inputs):
        return 1 / (1 + math.exp(-inputs))
