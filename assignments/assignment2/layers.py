import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    
    loss = reg_strength * (W ** 2).sum()
    grad = reg_strength * 2 * W

    return loss, grad


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    """
    
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    loss = 0.0
    for i in range(probs.shape[0]):
        loss -= np.log(probs[i][target_index[i]])
    return loss


def softmax(predictions):
    """
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    """
    
    soft_predictions = predictions.copy()
    if predictions.ndim == 1:
        soft_predictions -= np.max(soft_predictions)
        soft_predictions = np.exp(soft_predictions)
        return soft_predictions / np.sum(soft_predictions)
    for i in range(soft_predictions.shape[0]):
        soft_predictions[i] -= np.max(soft_predictions[i])
        soft_predictions[i] = np.exp(soft_predictions[i])
        soft_predictions[i] = soft_predictions[i] / np.sum(soft_predictions[i])
    return soft_predictions


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    
    soft_predictions = softmax(predictions)
    loss = cross_entropy_loss(soft_predictions, target_index)
    true_labels = np.zeros_like(soft_predictions)
    
    if predictions.ndim == 1:
        true_labels[target_index] = 1.0
        return loss, soft_predictions - true_labels
    for i in range(soft_predictions.shape[0]):
            true_labels[i][target_index[i]] = 1.0
    return loss / soft_predictions.shape[0], (soft_predictions - true_labels) / soft_predictions.shape[0]


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
        
    def clear_grad(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        res = X.copy()
        res[res < 0] = 0.0
        return res
    
    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        
        grad = np.where(self.X < 0, 0, 1)
        d_result = grad * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        
        self.X = Param(X)
        output = self.X.value @ self.W.value + self.B.value
        return output

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        
        self.W.grad += self.X.value.T @ d_out
        self.B.grad += d_out.sum(axis=0)
        
        d_input = d_out @ self.W.value.T
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
