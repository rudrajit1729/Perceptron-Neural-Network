import numpy as np

class NeuralNetwork():

	def __init__(self):
		#Seed the random number generator
		np.random.seed(1)
		# Set synaptic weights to a 3x1 matrix, with values (-1,1) and mean 0
		self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

	def sigmoid(self, x):
		'''
		Takes in weighted sum of the inputs and normalizes them
		between 0 and 1 through sigmoid function 
		'''
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		'''
		The derivative of the sigmoid function used to calculate necessary
		weight adjustments
		'''
		return x * (1-x)

	def train(self,training_inputs, training_outputs, training_iterations):
		'''
		We train the model through trail and error, adjusting the synaptic
		weights each time to get a better result

		'''

		for iteration in range(training_iterations):
			#Pass training set to the neural network
			outputs = self.think(training_inputs)

			#Calculate the error rate
			error = training_outputs - outputs

			# Multiply error by input and gradient of the sigmoid function
			# Less confident weights are adjusted more through the nature of the function
			adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(outputs))

			# Adjust the synaptic weights
			self.synaptic_weights += adjustments

	def think(self, inputs):
		# Pass inputs to neural network to get output

		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

		return output

if __name__ == "__main__":
	# Initialize the single neuron neural network
	neural_network = NeuralNetwork()

	print("Random synaptic weights: ")
	print(neural_network.synaptic_weights)
	
	# The training set, with 4 examples consisting of 3
	# input values and 1 output value
	training_inputs = np.array([[0, 0, 1],
							[1, 1, 1],
							[1, 0, 1],
							[0, 1, 1]])

	training_outputs = np.array([[0, 1, 1, 0]]).T

	# Train the neural network
	neural_network.train(training_inputs, training_outputs, 100000)

	print("Synaptic Weights after training: ")
	print(neural_network.synaptic_weights)

	# Take data from user
	A = str(input("Input 1: "))
	B = str(input("Input 2: "))
	C = str(input("Input 3: "))

	print("New Situation: input data = [{0}, {1}, {2}]".format(A, B, C))
	#Pass data to the neural network and get the output
	op = neural_network.think(np.array([A, B, C]))
	print("Ouput data = ", op)
	# Sigmoid function returns 1 when inf number of iterations done
	# We generate 1 as op if prediction is > 0.9
	if (op > 0.9):
		op = 1
	else:
		op = 0
	print("Prediction: ", op)

