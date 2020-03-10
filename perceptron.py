import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
							[1, 1, 1],
							[1, 0, 1],
							[0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# 3x1 matrix as 3 i/ps 1 o/p
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

#Training process
'''Take i/ps from training example
	Pass it on the netework
	Calculate error ---> depending on its severeness
						adjust the weight
	Error weighted derivative
	error = output - actual output
	Adjust weights by = error.input.sigmoid'(output)
	[sigmoid' = derivative of sigmoid function = y(1-y)]
	Repeat this process a large number of times
'''

for iteration in range(50000):
	input_layer = training_inputs
	
	outputs = sigmoid (np.dot(input_layer, synaptic_weights))#sigmoid function arameter-->weighted sum xiwi
	
	error = training_outputs - outputs
	
	adjustments = error*sigmoid_derivative(outputs)

	synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training")
print(synaptic_weights)

print("Outputs after training: ")
print(outputs)
