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

    def activation(self, tensor, func='relu'):
        if func=='linear':
            return tensor
        elif func=='relu':
            return np.where(tensor > 0, tensor, 0)
        elif func=='sigmoid':
            return 1/(1+np.exp(-tensor))
        elif func=='tanh':
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

            new_weight = np.add(self.weight, np.random.random(self.weight.shape))
            new_bias = np.add(self.bias, np.random.random(self.bias.shape))

            y_pred2 = np.matmul(x_input, new_weight)
            y_pred2 = np.add(y_pred2, new_bias)
            y_pred2 = self.activation(y_pred2)

            diff_new = self.loss(y_true, y_pred2)

            if diff_original > diff_new:
                self.weight = new_weight
                self.bias = new_bias
                print(f'loss: {diff_new}')
                # print(f'Epoch {epoch} - weight:{self.weight}, bias:{self.bias}, output:{y_pred2}')


if __name__ == '__main__':
    x = np.array([-10, 20, 0, 30, 4])
    y= np.array([2, 10])

    ann = ANN(5, 2)
    print(f'[initial] weight:{ann.weight}, bias:{ann.bias}, output:{ann(x)}')

    epochs=10000
    ann.minimize(x,y, epochs=epochs)
    print(f'[learned(epochs={epochs})] weight:{ann.weight}, bias:{ann.bias}, output:{ann(x)}')