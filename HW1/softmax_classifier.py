import numpy as np

def softmax_classifier(W, input, label, _lambda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - _lambda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N,), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    scores = np.dot(input, W) 
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    N = input.shape[0]
    correct_logprobs = -np.log(probs[range(N), np.argmax(label, axis=1)])
    data_loss = np.sum(correct_logprobs) / N
    reg_loss = 0.5 * _lambda * np.sum(W * W)
    loss = data_loss + reg_loss
    dscores = probs.copy()# must copy, otherwise probs will be changed
    dscores[range(N), np.argmax(label, axis=1)] -= 1
    dscores /= N
    gradient = np.dot(input.T, dscores) + _lambda * W
    prediction = np.argmax(probs, axis=1)

    ############################################################################

    return loss, gradient, prediction
