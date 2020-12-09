import numpy as np


class ANN(object):
    def __init__(self, input_shape, output_shape):
        self.weight = np.zeros(input_shape, output_shape)
        self.bias = np.ones(output_shape)

    def call(self, tensor):
        tensor = np.matmul(self.weight, tensor)
        tensor = np.add(tensor, self.bias)
        return tensor

    def activation(self, tensor):
        return tensor

    def minimize(self, label, pred):
        diff1 = np.abs(label-pred)
        diff1 = np.sum(np.square(diff1))

        random_value = np.random.random(pred.shape)
        random_added = np.add(pred, random_value)

        random_added = np.abs(label-random_added)
        diff2 = np.sum(random_added)

        if diff2 < diff1:
            self.minimize(label, pred)
        else:
            return random_added