from MLP import Layer, MLP
from Dataset import MultiClassDataset
from ActivationFunction import Logistic

import sys

#Primer argumento: Dimensión capa oculta
hidden_layer_dimension = int(sys.argv[1])
#Segundo argumento: Dimensión del output
output_layer_dimension = int(sys.argv[2])
#Tercer argumento: Nombre del archivo de error (Sin extensión)
error_file_name = sys.argv[3]

logistic = Logistic()
#Hidden Layer
layer_0 = Layer(
    dimension=hidden_layer_dimension,
    input_dimension=784,
    activation_function=logistic
)

layer_1 = Layer(
    dimension=output_layer_dimension,
    input_dimension=hidden_layer_dimension,
    activation_function=logistic
)

layer_list = [layer_0, layer_1]
classifier = MLP(layer_list)

dataset_train = MultiClassDataset('mnist_train.csv', dict([(str(i), i) for i in range(10)]), 0.8)
dataset_train.normalize_data(lambda x: x/255)

classifier.train_network(dataset_train, 50, 0.1, 0.9, True, error_file_name)

dataset_test = MultiClassDataset('mnist_test.csv', dict([(str(i), i) for i in range(10)]))
dataset_test.normalize_data(lambda x: x/255)
classifier.eval(dataset_test)


