
from numpy.core.defchararray import multiply
from metrics import accuracy, precision
import numpy as np
import random
import csv

"""La clase Perceptron representa un único perceptron"""
class Perceptron:

    """ Constructor de la clase perceptron. Inicializa aleatoriamente los pesos
        de cada input entre -0.05 y 0.05
        Parámetros:
            - input_dimension: Cantidad de inputs que recibe el perceptron (Sin contar el sesgo)
            - activation_function: Función de un parámetro que será usada como función de activación
    """
    def __init__(self, input_dimension, activation_function):

        self.weights = np.array([random.uniform(-0.05, 0.05) for i in range(input_dimension + 1)])
        self.activation_function = activation_function

    """ Suma pesada de los inputs:
        Parámetros:
            - inputs: Vector que actúa como input del perceptron
    """
    def sum_inputs(self, inputs):

        multiplication = self.weights * inputs
        return multiplication.sum()

    """ Aplica la función de activación a la suma pesada de los inputs
        Parámetros:
            - inputs: Vector que actúa como input del perceptron
    """
    def output(self, inputs):

        return self.activation_function.output(self.sum_inputs(inputs))

    """ Permite ajustar los pesos del perceptron cuando hay un dato mal clasificado 
        Parámetros:
            - expected_value: Valor que se esperaba devolviera el perceptron para el dato dado
            - output_value: Valor devuelto por el perceptron para el dato dado
            - learning_rate: Tasa de aprendizaje a aplicar
            - features: El dato a partir del cual se obtuvo el resultado de output_value
    """
    def adjust_weights(self, expected_value, output_value, learning_rate, features):

        factor = learning_rate * (expected_value - output_value)

        delta = factor * features

        self.weights = self.weights + delta

    """ Salva los pesos obtenidos en el último epoch en un archivo csv
        Parámetros:
            - filename: Nombre del archivo en el que se guardaran los pesos
    """
    def save_weights(self, filename):

        with open(f'{filename}.csv', 'w') as csvfile:

            writer = csv.writer(csvfile, delimiter=",")

            writer.writerow(self.weights)

            csvfile.close()


    """ Entrena un perceptron para clasificar los datos en un dataset
        Parámetros:
            - dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_weights: Nombre del archivo donde se guardaran los pesos. Si el nombre es el 
              string vacío no se salvan los pesos
    """
    def train_perceptron(self, dataset, epochs, learning_rate, verbose=False, save_weights=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.weights))

        print("Training information\n")
        print("Epoch, Precision, Accuracy")

        for current_epoch in range(epochs):

            epoch_errors = False
            error_number = 0
            true_positives = 0
            false_positives = 0

            for features, expected_value in dataset.training_data_iter():

                output_value = self.output(features)
                
                if output_value != expected_value:

                    epoch_errors = True
                    error_number += 1

                    if expected_value == 0:
                        false_positives +=1

                    self.adjust_weights(expected_value, output_value, learning_rate, features)
                else:
                    if expected_value == 1:
                        true_positives += 1


            if not epoch_errors:
                if verbose:
                    print(f'{current_epoch + 1}, {precision(true_positives, false_positives)}, {accuracy(dataset.training_data_size(), error_number)}')
                break
            else:
                if verbose:
                    print(f'{current_epoch + 1}, {precision(true_positives, false_positives)}, {accuracy(dataset.training_data_size(), error_number)}')

        if save_weights != "":
            self.save_weights()
        else:
            if verbose:
                print("Pesos finales: ")
                print(self.weights)
                print("")

    """ Devuelve la precision y la accuracy para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
    """
    def eval(self, dataset):

        print("Test information\n")
        print("Precision, Accuracy")

        error_number = 0
        true_positives = 0
        false_positives = 0

        for features, expected_value in dataset.test_data_iter():

            output_value = self.output(features)
            
            if output_value != expected_value:

                error_number += 1

                if expected_value == 0:
                    false_positives +=1

            else:
                if expected_value == 1:
                    true_positives += 1


        print(f'{precision(true_positives, false_positives)}, {accuracy(dataset.test_data_size(), error_number)}')



