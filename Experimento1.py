from MLP import Layer, MLP
from Dataset import MultiClassDataset
from ActivationFunction import Sigmoid

import sys

hidden_layer_dimension = int(sys.args[1])
output_layer_dimension = int(sys.args[2])

sigmoid = Sigmoid()
#Hidden Layer
layer_0 = Layer(
    dimension=hidden_layer_dimension,
    input_dimension=784,
    activation_function=sigmoid
)

layer_1 = Layer(
    dimension=output_layer_dimension,
    input_dimension=hidden_layer_dimension,
    activation_function=sigmoid
)

layer_list = [layer_0, layer_1]
classifier = MLP(layer_list)

dataset_train = MultiClassDataset('mnist_train.csv', dict([(str(i), i) for i in range(10)]))
dataset_train.normalize_data(lambda x: x/255)

classifier.train_network(dataset_train, 50, 0.1, True)

dataset_test = MultiClassDataset('mnist_test.csv', dict([(str(i), i) for i in range(10)]))
dataset_test.normalize_data(lambda x: x/255)
classifier.eval(dataset_test)


