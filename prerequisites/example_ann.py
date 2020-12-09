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

    def minimize(self, x_input, y_label, epochs=100):
        for epoch in range(epochs):
            diff_original = np.square(np.abs(y_label-self(x_input)))

            new_weight = np.add(self.weight, np.random.random(self.weight.shape))
            new_bias = np.add(self.bias, np.random.random(self.bias.shape))

            y_pred2 = np.matmul(x_input, new_weight)
            y_pred2 = np.add(y_pred2, new_bias)
            y_pred2 = self.activation(y_pred2)

            diff_new = np.square(np.abs(y_label - y_pred2))

            if diff_original > diff_new:
                self.weight = new_weight
                self.bias = new_bias
                # print(f'Epoch {epoch} - weight:{self.weight}, bias:{self.bias}, output:{y_pred2}')


if __name__ == '__main__':
    x = np.array([-10, 20, 0, 30, 4])
    y= np.array([0])

    ann = ANN(5, 1)
    print(f'[initial] weight:{ann.weight}, bias:{ann.bias}, output:{ann(x)}')

    epochs=100
    ann.minimize(x,y, epochs=epochs)
    print(f'[learned(epochs={epochs})] weight:{ann.weight}, bias:{ann.bias}, output:{ann(x)}')