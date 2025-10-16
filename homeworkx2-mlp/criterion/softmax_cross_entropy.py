import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

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
		# 保存logit和gt用于反向传播
		self.logit = logit
		self.gt = gt

		# 计算softmax
		exp_logits = np.exp(logit - np.max(logit, axis=1, keepdims=True))
		softmax = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + EPS)
		self.softmax = softmax

        # 交叉熵损失
		self.loss = -np.mean(np.sum(gt * np.log(softmax + EPS), axis=1))

        # 计算准确率
		pred = np.argmax(softmax, axis=1)
		label = np.argmax(gt, axis=1)
		self.acc = np.mean(pred == label)
	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		grad = (self.softmax - self.gt) / self.logit.shape[0]
		return grad
		# 经过数学计算后，一个非常优雅的结果就是多分类的softmax交叉熵损失的梯度就是预测概率减去真实标签。

	    ############################################################################
