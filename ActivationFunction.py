from abc import ABC, abstractmethod
import math

class ActivationFunction(ABC):

    @abstractmethod
    def output(self, x):
        pass

    @abstractmethod
    def first_derivative_output(self, x):
        pass

class Sigmoid(ActivationFunction):

    def __init__(self, alpha=1):
        self.alpha=alpha

    def output(self, x):
        return 1 / (1 + math.exp(-self.alpha * x))

    def first_derivative_output(self, x):
        output_val = self.output(x)
        return self.alpha * output_val * (1 - output_val)