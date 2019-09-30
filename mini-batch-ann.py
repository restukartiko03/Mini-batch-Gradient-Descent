import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class NeuralNetwork:
    def __init__(self, learning_rate=0.5, momentum=0, epochs=10, batch_size=10):
        self.weights = []
        self.biases = []
        self.layers_size = []
        self.activations = []
        self.previous_gradient = []
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        
    def add_hidden_layer(self, num_neuron):
        self.layers_size.append(num_neuron)
        self.activations.append(None)
        self.previous_gradient.append(0)
        if len(self.layers_size) == 1:
            return

        weight = np.random.randn(self.layers_size[-2], self.layers_size[-1])
        self.weights.append(weight)

        bias = np.zeros((1, num_neuron))
        self.biases.append(bias)        
    
    def add_input_layer(self, num_feature):
        weight = np.random.randn(num_feature, self.layers_size[0])
        self.weights = [weight] + self.weights

        bias = np.zeros((1, self.layers_size[0]))
        self.biases = [bias] + self.biases     

    def add_output_layer(self, num_output):
        self.add_hidden_layer(1)

    def feed_forward(self, features):
        self.activations[0] = sigmoid(np.dot(features, self.weights[0]) + self.biases[0])
        for i in range(1, len(self.layers_size)):
            self.activations[i] = sigmoid(np.dot(self.activations[i-1], self.weights[i]) + self.biases[i])
        
        
    def back_propagation(self, features, labels):
        print('Error :', error(self.activations[-1], labels))
        
        delta = [cross_entropy(self.activations[-1], labels)]
        for i in range(len(self.layers_size)-1, 0, -1):
            new_delta = [np.dot(delta[0], self.weights[i].T) * sigmoid_derv(self.activations[i-1])]
            delta = new_delta + delta

        for i in range(len(self.layers_size)-1, 0, -1):
            gradient = self.learning_rate * np.dot(self.activations[i-1].T, delta[i])
            self.weights[i] -= (gradient + self.momentum * self.previous_gradient[i])
            self.biases[i] -= self.learning_rate * np.sum(delta[i], axis=0)
            self.previous_gradient[i] = gradient

        gradient = self.learning_rate * np.dot(features.T, delta[0])
        self.weights[0] -= (gradient + self.momentum * self.previous_gradient[0])
        self.biases[0] -= self.learning_rate * np.sum(delta[0], axis=0)
        self.previous_gradient[0] = gradient

    def train(self, features, labels):
        self.add_hidden_layer(labels.shape[1])
        self.add_input_layer(features.shape[1])

        for i in range(self.epochs):
            print("epoch: " +  str(i))
            
            for j in range(int(math.ceil(len(features) / self.batch_size))): # numbers of batch
                first_index = j*self.batch_size
                batch = self.batch_size if len(features) - first_index >= self.batch_size else len(features) % self.batch_size
                
                data_train = features[first_index : first_index + batch]
                label_train = labels[first_index : first_index + batch]
              
                self.feed_forward(np.array(data_train))
                self.back_propagation(np.array(data_train), np.array(label_train))

    def predict(self, data):
        self.feed_forward(data)
        return self.activations[len(self.layers_size) - 1].argmax()
			

epochs = 300
learning_rate = 0.001
momentum = 0.1
batch_size = 15

model = NeuralNetwork(epochs=epochs, learning_rate=learning_rate, momentum=momentum, batch_size=batch_size)
model.add_hidden_layer(128)
model.add_hidden_layer(128)
model.train(x_train, y_train)
		
def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100
	
print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))
print("Test accuracy : ", get_acc(x_val/16, np.array(y_val)))