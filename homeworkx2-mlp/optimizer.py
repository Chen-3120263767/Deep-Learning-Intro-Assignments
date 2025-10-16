
import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	# One backpropagation step, update weights layer by layer
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:

				############################################################################
			    # TODO: Put your code here
				# Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
				# Do not forget the weightDecay term.
				layer.diff_W = -self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
				layer.diff_b = -self.learningRate * layer.grad_b
				# 权重衰减的本质其实是正则化，Loss_new = Loss_original + (λ / 2) * Σ(weights²)
				# w = w - learning_rate * dw,dw_new = dw_original + λ * w
				# 所以更新规则变为了w = (1 - learning_rate * λ) * w - learning_rate * dw_original
				# (1 - learning_rate * λ) 是一个小于1的数，相当于对权重进行了衰减
				# 偏置b一般不进行正则化
			    ############################################################################

				# Weight update
				layer.W += layer.diff_W
				layer.b += layer.diff_b
