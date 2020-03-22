import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        
        self.reg = reg
        
        self.input_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        for param in params:
            params[param].clear_grad()
            
        res = self.input_layer.forward(X)
        res = self.relu.forward(res)
        res = self.output_layer.forward(res)
        
        loss, dpred = softmax_with_cross_entropy(res, y)
        
        grad = self.output_layer.backward(dpred)
        grad = self.relu.backward(grad)
        grad = self.input_layer.backward(grad)
        
        for param in params:
            loss_l2, grad_l2 = l2_regularization(params[param].value, self.reg)
            loss += loss_l2
            params[param].grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        pred = self.input_layer.forward(X)
        pred = self.relu.forward(pred)
        pred = self.output_layer.forward(pred)
        
        pred = np.argmax(softmax(pred), axis=1)
        return pred
    
    def params(self):
        result = {'W_h': self.input_layer.W, 'B_h': self.input_layer.B, 'W_o': self.output_layer.W,
                  'B_o': self.output_layer.B}

        return result
