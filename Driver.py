import numpy as np
import Topology
import Network
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def fitness_callback(net, x_test, y_test):
    test_error = 0
    for j in range(len(x_test)):
        test_error += (net.predict(x_test[j]) - y_test[j]) ** 2
    test_error = np.sum(test_error) / len(x_test)
    return 1 - test_error

print("Iris dataset:")

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
top = Topology.Topology()
top.add_layer(4, "Input")
top.add_layer(5)
top.add_layer(5)
top.add_layer(3)
net = Network.Network(top, learning_rate=0.01)

plot_data = net.fit(x_train, y_train, x_test, y_test, 1000)

plt.plot(plot_data[0], plot_data[1], label="Train")
plt.plot(plot_data[0], plot_data[2], label="Test")
plt.xlabel = "Iteration"
plt.ylabel = "Error"
plt.title = "Diabetes"
plt.text = "Diabetes"
plt.legend()
plt.show()


print()

print("Diabetes dataset:")

diabetes = load_diabetes()

data = diabetes["data"]
targets = diabetes["target"]

data = np.transpose(data)
for index, column in enumerate(data):
    column_mean = np.mean(column)
    column_std = np.std(column)
    data[index] = (column - column_mean) / column_std

data = np.transpose(data)
data = np.ndarray.tolist(data)


target_mean = np.mean(targets)
target_std = np.std(targets)
targets = (targets - 25) / (346-25)
targets = np.ndarray.tolist(targets)
for i in range(len(targets)):
    targets[i] = [targets[i]]


x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)

top = Topology.Topology()
top.add_layer(10, "Input")
top.add_layer(6)
top.add_layer(1)
net = Network.Network(top, 0.01)


plot_data = net.fit(x_train, y_train, x_test, y_test, 1000)
plt.plot(plot_data[0], plot_data[1], label="Train")
plt.plot(plot_data[0], plot_data[2], label="Test")
plt.xlabel = "Iteration"
plt.ylabel = "Error"
plt.title = "Diabetes"
plt.text = "Diabetes"
plt.legend()
plt.show()

# The following is from a kaggle dataset. It shrinks the amount of data it has to sift through,
# and it still ran for about 90 minutes. Trained very badly.

# print()
#
# print("Santander:")
#
# # load train data
# print("Loading training data...")
# data = np.genfromtxt("train.csv", delimiter=',')
# data = np.delete(data, 0, axis=0)
# data = np.transpose(data)
# data = np.delete(data, 0, axis=0)
# x_train = np.delete(data, 0, axis=0)
# y_train = data[0]
# for index, column in enumerate(x_train):
#     column_mean = np.mean(column)
#     column_std = np.std(column)
#     x_train[index] = (column - column_mean) / column_std
# x_train = np.transpose(x_train)
# y_train = np.ndarray.tolist(y_train)
# for i in range(len(y_train)):
#     y_train[i] = [y_train[i]]
#
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.99)
# x_test, garbage, y_test, garbage = train_test_split(x_test, y_test, test_size=0.99)
#
# print("Creating the network...")
# top = Topology.Topology()
# top.add_layer(200, "Input")
# top.add_layer(100)
# top.add_layer(50)
# top.add_layer(25)
# top.add_layer(10)
# top.add_layer(5)
# top.add_layer(1)
# net = Network.Network(top,0.01)
# print("Fitting...")
# plot_data = net.fit(x_train, y_train, x_test, y_test, 100)
# plt.plot(plot_data[0], plot_data[1], label="Train")
# plt.plot(plot_data[0], plot_data[2], label="Test")
# plt.xlabel = "Iteration"
# plt.ylabel = "Error"
# plt.title = "Diabetes"
# plt.text = "Diabetes"
# plt.legend()
# plt.show()