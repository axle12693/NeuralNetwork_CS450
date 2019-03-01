import random
import numpy as np


class Network:
    def __init__(self, topology):
        self.topology = topology
        self.weights = []

        for layer_index in range(len(topology.layers) - 1):
            self.weights.append([])
            for k_neuron in range(topology.layers[layer_index + 1][1]):
                self.weights[layer_index].append([])
                self.weights[layer_index][k_neuron].append(random.uniform(-1.0, 1.0))
                for j_neuron in range(topology.layers[layer_index][1]):
                    self.weights[layer_index][k_neuron].append(random.uniform(-1.0, 1.0))

        print()
    def predict(self, data):
        next_layer_input = np.array(data)
        for index in range(len(self.topology.layers)-1):
            next_layer_input = np.concatenate(([-1], next_layer_input))
            next_layer_input = np.array(self.weights[index]).dot(next_layer_input)
            next_layer_input = self.topology.layers[index+1][0](next_layer_input)
        return next_layer_input




