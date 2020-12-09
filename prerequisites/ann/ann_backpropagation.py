import numpy as np


class ANN(object):
    def __init__(self, input_shape, output_shape):
        self.weight = np.zeros([input_shape, output_shape])
        self.bias = np.ones(output_shape)

    def __call__(self, tensor):
        tensor = np.matmul(tensor, self.weight)
        tensor = np.add(tensor, self.bias)
        tensor = self.activation(tensor)
        return tensor

    def activation(self, tensor):
        return np.where(tensor > 0, tensor, 0)

    def loss(self, y_true, y_pred):
        tensor = np.abs(y_true - y_pred)
        tensor = np.square(tensor)
        tensor = np.sum(np.squeeze(tensor))
        return tensor

    def minimize(self, x_input, y_true, epochs=100):
        # back-propagation
        for epoch in range(epochs):
            y_pred = self(x_input)
            loss = self.loss(y_true, y_pred)
            # back-propagation