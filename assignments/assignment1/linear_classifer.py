import numpy as np


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
        

def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
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
    

def linear_softmax(X, W, target_index):
    """
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    """

    predictions = X @ W

    loss, dW = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T @ dW
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        """
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        """

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_indices in batches_indices:
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                loss, grad = linear_softmax(X_batch, self.W, y_batch)
                loss_l2, grad_l2 = l2_regularization(self.W, reg)
                loss += loss_l2
                grad += grad_l2
                loss_history.append(loss)
                self.W -= learning_rate * grad
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        """
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        y_pred = softmax(np.dot(X, self.W))
        return y_pred.argmax(axis=1)
