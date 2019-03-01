import random


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
        for layer in self.topology.layers:
            pass




