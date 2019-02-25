import random


class Network:
    def __init__(self, topology):
        self.topology = topology
        self.weights = [[[random.uniform(-1.0, 1.0) for j_neuron in range(topology.layers[layer_index][1])] for k_neuron in range(topology.layers[layer_index + 1][1])] for layer_index in range(len(topology.layers) - 1)]

    def predict(self, data):
        for layer in self.topology.layers:
            pass
