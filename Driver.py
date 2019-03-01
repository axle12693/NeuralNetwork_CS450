# import numpy as np
#
# def calc(ls):
#     neuron_output_layer_1 = np.array(ls)
#     weights_layer1 = np.array([[1,3],[2,4]])
#     neuron_input_layer_2 = weights_layer1.dot(neuron_output_layer_1)
#     print(neuron_input_layer_2)
#     neuron_output_layer_2 = np.maximum(neuron_input_layer_2, 0)
#     weights_layer_2 = np.array([[5, 6]])
#     neuron_input_layer_3 = weights_layer_2.dot(neuron_output_layer_2)
#     neuron_output_layer_3 = np.maximum(neuron_input_layer_3, 0)
#
#     return neuron_output_layer_3
#
#
# print(calc([1,5]))


import Topology
import Network
top = Topology.Topology()
top.add_layer(2, "Input")
top.add_layer(2)
top.add_layer(1)
net = Network.Network(top)
