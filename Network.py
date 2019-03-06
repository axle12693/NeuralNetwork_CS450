import random
import numpy as np
from copy import deepcopy


class Network:
    def __init__(self, topology, learning_rate=0.1):
        self.topology = topology
        self.weights = []
        self.activations = []
        self.errors = []
        self.learning_rate = learning_rate

        for layer_index in range(len(topology.layers) - 1):
            self.weights.append([])
            for k_neuron in range(topology.layers[layer_index + 1][1]):
                self.weights[layer_index].append([])
                self.weights[layer_index][k_neuron].append(random.uniform(-1.0, 1.0))
                for j_neuron in range(topology.layers[layer_index][1]):
                    self.weights[layer_index][k_neuron].append(random.uniform(-1.0, 1.0))
        self.weights_copy = deepcopy(self.weights)  # enables us to avoid altering the weights until after all neuron
                                                    # errors are calculated

    def predict(self, row):
        self.activations = [row]
        next_layer_input = np.array(row)
        for index in range(len(self.topology.layers)-1):
            next_layer_input = np.concatenate(([-1], next_layer_input))
            next_layer_input = np.array(self.weights[index]).dot(next_layer_input)
            next_layer_input = self.topology.layers[index+1][0](next_layer_input)
            self.activations.append(next_layer_input)

        return next_layer_input

    def fit(self, data, targets, num_epochs=100):
        for i in range(num_epochs):
            print("Epoch " + str(i))
            for index, row in enumerate(data):
                self.errors = []
                self.predict(row)
                output_layer_calculated = False
                for neuron_layer_index in range(len(self.activations)-1, -1, -1):
                    if not output_layer_calculated:
                        self.calc_errors_and_update_weights(neuron_layer_index, targets[index])
                        output_layer_calculated = True
                    else:
                        self.calc_errors_and_update_weights(neuron_layer_index)
                self.weights = deepcopy(self.weights_copy)

    def calc_errors_and_update_weights(self, neuron_layer_index, targets=[]):
        """Update weights feeding into the referenced layer of neurons."""
        if neuron_layer_index == 0:
            return
        if len(targets) > 0: #if neuron_layer_index is the output layer
            neuron_layer_errors = []
            for j_neuron_index in range(len(self.activations[neuron_layer_index])):
                activation = self.activations[neuron_layer_index][j_neuron_index]
                error = activation * (1 - activation) * (activation - targets[j_neuron_index])
                neuron_layer_errors.append(error)
                for i_neuron_index in range(len(self.activations[neuron_layer_index - 1])):
                    i_neuron_activation = self.activations[neuron_layer_index - 1][i_neuron_index]
                    amount_to_change_weight = self.learning_rate * error * i_neuron_activation
                    self.weights_copy[neuron_layer_index - 1][j_neuron_index][i_neuron_index] -= amount_to_change_weight
                    #print(amount_to_change_weight)
            self.errors = [neuron_layer_errors] + self.errors
        else:
            neuron_layer_errors = []
            for j_neuron_index in range(len(self.activations[neuron_layer_index])):
                activation = self.activations[neuron_layer_index][j_neuron_index]
                weighted_sum_of_previous_errors = 0
                for k_neuron_index in range(len(self.activations[neuron_layer_index + 1])):
                    weighted_sum_of_previous_errors += self.errors[0][k_neuron_index]*self.weights[neuron_layer_index][k_neuron_index][j_neuron_index]
                error = activation * (1 - activation) * weighted_sum_of_previous_errors
                neuron_layer_errors.append(error)
                for i_neuron_index in range(len(self.activations[neuron_layer_index - 1])):
                    i_neuron_activation = self.activations[neuron_layer_index - 1][i_neuron_index]
                    amount_to_change_weight = self.learning_rate * error * i_neuron_activation
                    self.weights_copy[neuron_layer_index - 1][j_neuron_index][i_neuron_index] -= amount_to_change_weight
                    #print(amount_to_change_weight)
            self.errors = [neuron_layer_errors] + self.errors

