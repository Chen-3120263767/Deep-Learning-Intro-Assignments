import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.
		self.Input = Input
		self.Output = 1 / (1 + np.exp(-Input))  # sigmoid函数
		return self.Output

	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta
		# sigmoid的导数：output * (1 - output)
		return delta * self.Output * (1 - self.Output)

	    ############################################################################
