import numpy as np
import random

np.random.seed(1)
class Neural_Net(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(self.sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        a= np.array(a, ndmin=2).T
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descent(self, dataset, epochs, eta=1): #eta == learning_rate
        for i in range(epochs):
            self.update_training_data(dataset, eta)
            print("Epoch {0} completed.".format(i))

    def update_training_data(self, dataset, eta):
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in dataset:
            delta_change_b, delta_change_w = self.back_propagation(x, y)
            change_b = [cb + dcb  for cb, dcb in zip(change_b, delta_change_b)]
            change_w = [cw + dcw for cw, dcw in zip(change_w, delta_change_w)] 
        
        self.biases = [b + eta * cb for b, cb in zip(self.biases, change_b)]
        self.weights = [w + eta * cw for w, cw in zip(self.weights, change_w)]

    def back_propagation(self, x, y):
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        x= np.array(x, ndmin=2).T
        y= np.array(y, ndmin=2).T
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backpropagation
        delta = self.cost_derivative(y, activations[-1]) * sigmoid_prime(zs[-1])
        change_b[-1] = delta
        change_w[-1] = np.dot(delta, activations[-2].T)
        
        for j in range(2, self.num_layers):
            z = zs[-j]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-j+1].T, delta) * sp
            change_b[-j] = delta
            change_w[-j] = np.dot(delta, activations[-j-1].T)
        return (change_b, change_w)

    def cost_derivative(self, y, output_activations):
        return (y - output_activations)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
y = np.array([0, 1, 1, 0])

dataset = list(zip(X, y))

net = Neural_Net([2,4,1])
net.gradient_descent(dataset, 1000)

result = net.feedforward( [0, 0] )

print("Result: {0}".format(result))
