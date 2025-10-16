import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.logit = logit
		self.gt = gt
		# 计算均方误差损失
		self.loss = np.mean(np.sum((logit - gt) ** 2, axis=1)) / 2

        # 计算准确率
		pred = np.argmax(logit, axis=1)
		label = np.argmax(gt, axis=1)
		self.acc = np.mean(pred == label)

	    ############################################################################

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		# dL/dlogit = (logit - gt)
		return (self.logit - self.gt) / self.logit.shape[0]

	    ############################################################################
