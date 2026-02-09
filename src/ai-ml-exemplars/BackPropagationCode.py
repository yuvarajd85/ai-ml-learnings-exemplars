'''
Created on 2/7/26 at 11:28 AM
By yuvarajdurairaj
Module Name BackPropagationCode
'''


import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# output for the given equation
def get_output(x, y, z):
    return x * x + 4 * y + z


# Generate data set
data = np.random.randint(1, 5, size=(10))
df = pd.DataFrame(columns=['X', 'Y', 'Z'])
df['X'] = data
df['Y'] = data
df['Z'] = data

df['output'] = df.apply(lambda k: get_output(k['X'], k['Y'], k['Z']), axis=1)

inputs = df[['X', 'Y', 'Z']]
inputs = inputs.values
e = np.array(df['output'])
expected_output = []
for i in e:
    expected_output.append([i])
expected_output = np.array(expected_output)

epochs = 10000
lr = 0.1
input_layer_neuron, hidden_layer_neuron, output_layer_neuron = 3, 3, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(input_layer_neuron, hidden_layer_neuron))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neuron))
output_weights = np.random.uniform(size=(hidden_layer_neuron, output_layer_neuron))
output_bias = np.random.uniform(size=(1, output_layer_neuron))

# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Output from neural network after {} epochs:\n {}".format(epochs, predicted_output))
