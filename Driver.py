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

import numpy as np

import Topology
import Network
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def print_error(net, x_test, y_test):
    total_error = 0
    n = 0
    for i in range(len(x_test)):
        total_error += int(np.argmax(net.predict(x_test[i])) != np.argmax(y_test[i]))
        n += 1
    error = total_error / n

    print("Error: " + str(error))

print("Loading and prepping data...")

iris = load_iris()
data = iris["data"]

data = np.transpose(data)
for index, column in enumerate(data):
    column_mean = np.mean(column)
    column_std = np.std(column)
    data[index] = (column - column_mean) / column_std

data = np.transpose(data)
data = np.ndarray.tolist(data)

targets = iris["target"]
targets_lists = []
for target in targets:
    target_list = [0, 0, 0]
    target_list[target] = 1
    targets_lists.append(target_list)
x_train, x_test, y_train, y_test = train_test_split(data, targets_lists, test_size=0.3)
print("Creating network...")
top = Topology.Topology()
top.add_layer(4, "Input")
top.add_layer(5)
top.add_layer(5)
top.add_layer(5)
top.add_layer(4)
top.add_layer(3)
net = Network.Network(top, learning_rate=0.01)
print("training network...")

print_error(net, x_test, y_test)

net.fit(x_train, y_train, 1000)

print_error(net, x_test, y_test)

print()


for i in range(len(x_test)):
    print("My prediction: " + str(net.predict(x_test[i])))
    print("Target: " + str(y_test[i]))