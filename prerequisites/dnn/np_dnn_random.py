import numpy as np
from copy import deepcopy


class DNN(object):
    def __init__(self, input_shape, output_shape, hidden_dim):
        self.weights = {
            'layer1': np.random.normal(size=[input_shape, hidden_dim]),
            'layer2': np.random.normal(size=[hidden_dim, output_shape]),
        }
        self.bias = {
            'layer1': np.random.normal(size=hidden_dim),
            'layer2': np.random.normal(size=output_shape),
        }

    def __call__(self, tensor, weights=None, bias=None):
        weights = self.weights if not weights else weights
        bias = self.bias if not bias else bias

        tensor = np.matmul(tensor, weights['layer1'])
        tensor = np.add(tensor, bias['layer1'])
        tensor = self.activation(tensor, func='relu')

        tensor = np.matmul(tensor, weights['layer2'])
        tensor = np.add(tensor, bias['layer2'])
        tensor = self.activation(tensor, func='relu')

        return tensor

    def activation(self, tensor, func='relu'):
        if func == 'linear':
            return tensor
        elif func == 'relu':
            return np.where(tensor > 0, tensor, 0)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-tensor))
        elif func == 'tanh':
            return np.tanh(tensor)
        else:
            raise KeyError('정의되지 않은 함수입니다.')

    def loss(self, y_true, y_pred):
        tensor = np.abs(y_true - y_pred)
        tensor = np.square(tensor)
        tensor = np.sum(np.squeeze(tensor))
        return tensor

    def minimize(self, x_input, y_true, epochs=100):
        # randomly minimizing
        for epoch in range(epochs):
            diff_original = self.loss(y_true, self(x_input))
            new_weights, new_bias = deepcopy(self.weights), deepcopy(self.bias)
            for key in new_weights:
                new_weights[key] = np.add(new_weights[key], np.random.normal(size=new_weights[key].shape)*np.random.randint(0,10, size=new_weights[key].shape))
                new_bias[key] = np.add(new_bias[key], np.random.normal(size=new_bias[key].shape)*np.random.randint(0,10, size=new_bias[key].shape))

            y_pred2 = self(tensor=x_input, weights=new_weights, bias=new_bias)

            diff_new = self.loss(y_true, y_pred2)
            if diff_original > diff_new:
                self.weights = new_weights
                self.bias = new_bias
                print(f'loss: {diff_new}')
                # print(f'Epoch {epoch} - weight:{self.weights}, bias:{self.bias}, output:{y_pred2}')


if __name__ == '__main__':
    x = np.array([-10, 20, 0, 30, 4])
    y = np.array([2, 10])

    dnn = DNN(input_shape=5, output_shape=2, hidden_dim=1)
    print(f'[initial] weight:{dnn.weights}, bias:{dnn.bias}, output:{dnn(x)}')

    epochs = 10000
    dnn.minimize(x, y, epochs=epochs)
    print(f'[learned(epochs={epochs})] weight:{dnn.weights}, bias:{dnn.bias}, output:{dnn(x)}')
