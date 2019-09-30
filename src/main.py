import numpy as np
from mini_batch_nn import NeuralNetwork
import matplotlib.pyplot as plt

def get_data():
    x_train = []
    x_train.append([1, 0, 0, 85, 85, 0])
    x_train.append([1, 0, 0, 80, 90, 1])
    x_train.append([0, 1, 0, 83, 86, 0])
    x_train.append([0, 0, 1, 70, 96, 0])
    x_train.append([0, 0, 1, 68, 80, 0])
    x_train.append([0, 0, 1, 65, 70, 1])
    x_train.append([0, 1, 0, 64, 65, 1])
    x_train.append([1, 0, 0, 72, 95, 0])
    x_train.append([1, 0, 0, 69, 70, 0])
    x_train.append([0, 0, 1, 75, 80, 0])
    
    x_val = []
    x_val.append([1, 0, 0, 75, 70, 1])
    x_val.append([0, 1, 0, 72, 90, 1])
    x_val.append([0, 1, 0, 81, 75, 0])
    x_val.append([0, 0, 1, 71, 91, 1])

    return np.asarray(x_train), np.asarray(x_val)

def get_labels():
    y_train = []
    y_train.append([0])
    y_train.append([0])
    y_train.append([1])
    y_train.append([1])
    y_train.append([1])
    y_train.append([0])
    y_train.append([1])
    y_train.append([0])
    y_train.append([1])
    y_train.append([1])

    y_val = []
    y_val.append([1])
    y_val.append([1])
    y_val.append([1])
    y_val.append([0])

    return np.asarray(y_train), np.asarray(y_val)

def plot_errors(errors):
    plt.plot(errors)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.show()

epochs = 100
learning_rate = 0.001
momentum = 0.1
batch_size = 5
x_train, x_val = get_data()
y_train, y_val = get_labels()

model = NeuralNetwork(epochs=epochs, learning_rate=learning_rate, momentum=momentum, batch_size=batch_size)
model.add_hidden_layer(128)
model.add_hidden_layer(128)
model.train(x_train, y_train)

print("Training accuracy : ", model.get_accuration(x_train, np.array(y_train)))
print("Test accuracy : ", model.get_accuration(x_val, np.array(y_val)))
plot_errors(model.errors)