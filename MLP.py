from Perceptron import Perceptron
from metrics import precision, accuracy, sample_error
import csv
import numpy as np
from utils import round_output

"""La clase Layer representa una capa de perceptrones. Esta muy sencilla implmentación asume
que todos los perceptrones de la capa reciben exactamente el mismo input"""
class Layer:

    """ Constructor de la clase Layer. 
        Parámetros:
            - dimension: Cantidad de perceptrones en la capa
            - input_dimension: Dimensiones del vector de input que recibirán los perceptrones de la capa
            - activation_function: Función de un solo parámetro que será usada como función de activación
              de los perceptrones
            - perceptron_list: En caso de no pasar ninguno de los parámetros anteriores, se presenta la
              opción de pasar directamente una lista de perceptrones. Nótese que si este parámetro se
              pasa junto con los demás, los demás serán ignorados
    
    """
    def __init__(self, dimension=None, input_dimension=None, activation_function=None, perceptron_list=[]):
        if perceptron_list == []:
            self.dimension = dimension
            self.neurons = np.array([Perceptron(input_dimension, activation_function) for i in range(dimension)])
            self.last_input = np.zeros(input_dimension + 1)
        else:
            self.dimension = len(perceptron_list)
            self.neurons = np.array(perceptron_list)
            self.last_input = np.zeros(len(self.neurons[0].weights))

    """ Aplica la función de activación a todos los perceptrones de la capa dado un dato y devuelve
        un vector (Representado con una lista) que contiene los resultados de cada perceptron
        Parámetros:
            - input_vector: Dato de entrada suministrado a la capa
    
    """
    def output(self, input_vector):

        self.last_input = input_vector

        return np.array([perceptron.output(input_vector) for perceptron in self.neurons])

    """
        Esta función se utiliza para el cálculo del gradiente local de una neurona que no está en la
        última capa de la MLP. Este método en particular calcula la parte de la sumatoria de
        gradientes por pesos
        Parámetros:
            - neuron_id: Identificador de la neurona en su respectiva capa (Índice dentro de la lista
            de neuronas)
    """
    def weighted_gradient_sum(self, neuron_id):

        return sum([neuron.localGradient * neuron.weights[neuron_id + 1] for neuron in self.neurons])

    """ Entrena una capa (Esto se utiliza en clasificadores de una sola capa, como en la pregunta 4)
        para clasificar un dataset
        Parámetros:
            - dataset: Una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_weights: Nombre del archivo donde se guardaran los pesos. Si el nombre es el 
              string vacío no se salvan los pesos (Esta parte no está funcionando porque no he
              implementado save_weights aquí)
    """
    def train_layer(self, dataset, epochs, learning_rate, verbose=False, save_weights=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.neurons[0].weights))

        labels_header = ",".join(["prec. label " + str(key) for key in dataset.get_labels()])       
        print("Training information\n")
        print(f'epoch, accuracy, {labels_header}')

        for current_epoch in range(epochs):

            epoch_errors = False
            error_number = 0
            true_positives = {}
            false_positives = {}

            for key in dataset.get_labels():
                true_positives[key] = 0
                false_positives[key] = 0

            for features, expected_value in dataset:

                output_value = self.output(features)

                index = dataset.get_label_index(expected_value)

                is_incorrect = False

                for i in range(len(output_value)):

                    if i == index:
                        if output_value[index] != 1:
                            is_incorrect = True
                            self.neurons[index].adjust_weights(1, -1, learning_rate, features)
                    else:
                        if output_value[i] != -1:
                            is_incorrect = True
                            self.neurons[i].adjust_weights(-1, 1, learning_rate, features)

                            if output_value.sum() == -8:
                                false_positives[str(i)] += 1

                if is_incorrect:
                    epoch_errors = True
                    error_number += 1
                else:
                    true_positives[str(index)] += 1

            precision_list = []

            for key in dataset.get_labels():
                precision_list.append(round(precision(true_positives[key], false_positives[key]), 4))

            precision_string = ",".join([str(value) for value in precision_list])

            if not epoch_errors:
                if verbose:
                    print(f'{current_epoch}, {accuracy(dataset.size(), error_number)}, {precision_string}')
                break
            else:
                if verbose:
                    print(f'{current_epoch}, {accuracy(dataset.size(), error_number)}, {precision_string}')
                    dataset.shuffle_all()

    """ Devuelve la precision y la accuracy para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
    """
    def eval(self, dataset):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.neurons[0].weights))

        labels_header = ",".join(["prec. label " + str(key) for key in dataset.get_labels()])
        print('Test information\n')
        print(f'accuracy, {labels_header}')

        error_number = 0
        true_positives = {}
        false_positives = {}

        for key in dataset.get_labels():
            true_positives[key] = 0
            false_positives[key] = 0


        for features, expected_value in dataset:

            output_value = self.output(features)

            index = dataset.get_label_index(expected_value)

            is_incorrect = False

            for i in range(len(output_value)):

                if i == index:
                    if output_value[index] != 1:
                        is_incorrect = True
                else:
                    if output_value[i] != -1:
                        is_incorrect = True

                        if output_value.sum() == -8:
                            false_positives[str(i)] += 1

            if is_incorrect:
                error_number += 1
            else:
                true_positives[str(index)] += 1

        precision_list = []

        for key in dataset.get_labels():
            precision_list.append(round(precision(true_positives[key], false_positives[key]), 2))

        print("ERROR NUMBER")
        print(error_number)
        print("SIZE")
        print(dataset.size())
        precision_string = ",".join([str(value) for value in precision_list])
        print(f'{accuracy(dataset.size(), error_number)}, {precision_string}')

        

"""Esta clase representa un multilayer perceptron"""
class MLP:

    """ Constructor de la clase MLP
        Parámetros:
            - layer_list: Una lista de objectos de la clase Layer. Nótese que las capas deben aparecer
              en la lista en el orden que se desea se apliquen
    """
    def __init__(self, layer_list):

        self.layers = np.array(layer_list)
        self.depth = len(layer_list)
        self.error_vector = np.zeros(self.layers[self.depth - 1].dimension)
        self.learning_rate = 0

    """ Genera el output del MLP. Funciona de la siguiente manera: Se itera por las capas de la red,
        en la primera iteración se recibe directamente el dato a clasificar, luego el output de esa 
        capa se utiliza como el input de la siguiente capa y así sucesivamente. Devuelve el output que
        da la última capa después de finalizar este proceso
        Parámetros:
            - input_vector: Vector que representa el dato a clasificar
    """
    def output(self, input_vector):

        next_layer_input = input_vector

        for layer_id in range(len(self.layers)):

            if layer_id != self.depth - 1:
                next_layer_input = np.concatenate((np.ones(1), self.layers[layer_id].output(next_layer_input)))
            else:
                next_layer_input = self.layers[layer_id].output(next_layer_input)

        return next_layer_input

    """ Calcula el gradiente local para una neurona específica
        Parámetros:
            - layer_id: Identificador de la capa donde está la neurona cuyo gradiente local se desea
            calcular
            - neuron_id: Identificador dentro de la capa de la neurona cuyo gradiente local se desea
            calcular
    """
    def setLocalGradient(self, layer_id, neuron_id):

        neuron_input = self.layers[layer_id].last_input
        
        if layer_id == (self.depth - 1):
            self.layers[layer_id].neurons[neuron_id].localGradient = self.error_vector[neuron_id] * self.layers[layer_id].neurons[neuron_id].first_derivative_output(neuron_input)
        else:
            sum_next_layer = self.layers[layer_id + 1].weighted_gradient_sum(neuron_id)  
            self.layers[layer_id].neurons[neuron_id].localGradient = self.layers[layer_id].neurons[neuron_id].first_derivative_output(neuron_input) * sum_next_layer

    """ Retorna el gradiente local de una neurona específica
        Parámetros:
            - layer_id: Identificador de la capa donde está la neurona cuyo gradiente local se desea
            calcular
            - neuron_id: Identificador dentro de la capa de la neurona cuyo gradiente local se desea
            calcular
    """
    def getLocalGradient(self, layer_id, neuron_id):

        return self.layers[layer_id].neurons[neuron_id].localGradient

    """ Algoritmo de backpropagation (Sin momentum) para una MLP (Fully connected y con sesgo)
        Parámetros:
            - layer_id: Capa cuyos pesos se ajustan en esta llamada a backpropagation
    """
    def backpropagation(self, layer_id):

        layer_inputs = self.layers[layer_id].last_input #último input que recibió la capa

        for neuron_id in range(self.layers[layer_id].dimension): #Para cada neurona en la capa

            self.setLocalGradient(layer_id, neuron_id) #Se calcula el gradiente local
            local_gradient = self.getLocalGradient(layer_id, neuron_id)

            delta = self.learning_rate * local_gradient * layer_inputs #Se calcula el delta de los pesos

            self.layers[layer_id].neurons[neuron_id].weights += delta #Se ajustan los pesos

        if layer_id != 0: #Si esta no es la primera capa, el algoritmo continúa en la capa anterior
            self.backpropagation(layer_id - 1)

    """ Algoritmo de backpropagation con momentum para una MLP (Fully connected y con sesgo)
        Parámetros:
            - layer_id: Capa cuyos pesos se ajustan en esta llamada a backpropagation
    """
    def backpropagation_with_momentum(self, layer_id):

        layer_inputs = self.layers[layer_id].last_input #último input que recibió la capa

        for neuron_id in range(self.layers[layer_id].dimension): #Para cada neurona en la capa

            self.setLocalGradient(layer_id, neuron_id) #Se calcula el gradiente local
            local_gradient = self.getLocalGradient(layer_id, neuron_id)

            delta = local_gradient * layer_inputs #Se calcula el delta de los pesos (Sin multiplicar por la tasa de aprendizaje)
            momentum = self.layers[layer_id].neurons[neuron_id].momentum #Se obtiene el momentum en esta iteración

            self.layers[layer_id].neurons[neuron_id].weights += (self.learning_rate * (delta + momentum)) #Se actualizan los pesos
            self.layers[layer_id].neurons[neuron_id].update_momentum(self.alpha, delta) #Se actualiza el momentum

        if layer_id != 0: #Si esta no es la primera capa, el algoritmo continúa en la capa anterior
            self.backpropagation(layer_id - 1)

    """ Entrena una capa una MLP para clasificar un dataset
        Parámetros:
            - dataset: Una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - alpha (Opcional): Factor de escalamiento del momentum
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_weights: Nombre del archivo donde se guardaran los pesos. Si el nombre es el 
              string vacío no se salvan los pesos (Esta parte no está funcionando porque no he
              implementado save_weights aquí)
    """
    def train_network(self, dataset, epochs, learning_rate, alpha=0, verbose=False, save_error=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.layers[0].neurons[0].weights))
        self.learning_rate = learning_rate

        if (alpha != 0):
            
            self.alpha = alpha

        labels_header = ",".join(["prec. label " + str(key) for key in dataset.get_labels()])       
        print("Training information\n")
        print(f'epoch, type, accuracy, MSE, {labels_header}')
        #print('epoch, MSE')

        if save_error != "":
            error_train_list =[['epoch', 'MSE']]
            error_val_list = [['epoch', 'MSE']]

        prev_mse = 0 #Aquí se guarda el mse de la epoch anterior

        for current_epoch in range(epochs):

            error_number = 0
            true_positives = {}
            false_positives = {}

            for key in dataset.get_labels():
                true_positives[key] = 0
                false_positives[key] = 0

            sum_mse = 0 #Aquí se va acumulando el error para cada muestra

            for features, expected_value in dataset.training_data_iter(): #Se itera sobre las muestras en el dataset

                output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)

                rounded_output = np.array([round_output(x) for x in output_value]) #Se redondean x > 0.9 como 1 y x < 0.1 como  0 para hacer la predicción

                expected_vector = dataset.get_label_vector(expected_value)

                if not all(rounded_output == expected_vector):
                    error_number += 1

                    if sum(rounded_output) == 1:
                        false_positives[expected_value] += 1
                else:
                    true_positives[expected_value] += 1

                error = sample_error(expected_vector, output_value) #Se calcula el error para la muestra

                self.error_vector = expected_vector - output_value
                if alpha == 0:
                    self.backpropagation(self.depth - 1)
                else:
                    self.backpropagation_with_momentum(self.depth - 1)

                sum_mse += error #Actualizar error total

            mse = sum_mse / dataset.training_data_size() #Calcular error promedio

            precision_list = []

            for key in dataset.get_labels():
                precision_list.append(round(precision(true_positives[key], false_positives[key]), 2))

            precision_string = ",".join([str(value) for value in precision_list])
            print(f'{current_epoch}, train, {accuracy(dataset.training_data_size(), error_number)}, {mse}, {precision_string}')

            if save_error != "":
                error_train_list.append([f'{current_epoch}, {mse}'])

            #Se aplica la validación luego del epoch
            self.validation(dataset, current_epoch, error_val_list)

            prev_mse = mse 
            dataset.shuffle_training_data() #Cambiar el orden en que se muestran los datos


        if save_error != '': #Escribir en un archivo el error cometido en cada epoch

            with open(f'{save_error}_train.csv', 'w') as training_results:
                writer = csv.writer(training_results)

                for row in error_train_list:
                    writer.writerow(row)

                training_results.close()

            with open(f'{save_error}_val.csv', 'w') as validation_results:
                writer = csv.writer(validation_results)

                for row in error_val_list:
                    writer.writerow(row)

                validation_results.close()

    """ Devuelve la precision, el mse y la accuracy para un dataset test. La única diferencia con
        eval es que este no añade sesgo al dataset (Se supone que ya viene incluido) y que registra
        el MSE en una lista. Se corre sobre una partición del dataset de entrenamiento
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - current_epoch: Epoch después de la cual se está ejecutando la validación
            - error_val_list: Donde se guardan el MSE de cada llamada a validación
    """
    def validation(self, dataset, current_epoch, error_val_list=[]):

        sum_mse = 0 #Aquí se va acumulando el error para cada muestra
        error_number = 0
        true_positives = {}
        false_positives = {}

        for key in dataset.get_labels():
            true_positives[key] = 0
            false_positives[key] = 0

        for features, expected_value in dataset.validation_data_iter(): #Se itera sobre las muestras en el dataset

            output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)

            rounded_output = np.array([round_output(x) for x in output_value])

            expected_vector = dataset.get_label_vector(expected_value)

            if not all(rounded_output == expected_vector):
                error_number += 1

                if sum(rounded_output) == 1:
                    false_positives[expected_value] += 1
            else:
                true_positives[expected_value] += 1


            error = sample_error(expected_vector, output_value) #Se calcula el error para la muestra

            sum_mse += error #Actualizar error total

        mse = sum_mse / dataset.validation_data_size() #Calcular error promedio

        precision_list = []

        if error_val_list != []:
            error_val_list.append([f'{current_epoch}, {mse}'])

        for key in dataset.get_labels():
            precision_list.append(round(precision(true_positives[key], false_positives[key]), 2))

        precision_string = ",".join([str(value) for value in precision_list])
        print(f'{current_epoch}, val, {accuracy(dataset.validation_data_size(), error_number)}, {mse}, {precision_string}')



    """ Devuelve la precision y la accuracy para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
    """
    def eval(self, dataset):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.layers[0].neurons[0].weights))


        labels_header = ",".join(["prec. label " + str(key) for key in dataset.get_labels()])       
        print("Test information\n")
        print(f'accuracy, mse, {labels_header}')

        sum_mse = 0 #Aquí se va acumulando el error para cada muestra
        error_number = 0
        true_positives = {}
        false_positives = {}

        for key in dataset.get_labels():
            true_positives[key] = 0
            false_positives[key] = 0

        for features, expected_value in dataset: #Se itera sobre las muestras en el dataset

            output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)

            rounded_output = np.array([round_output(x) for x in output_value])

            expected_vector = dataset.get_label_vector(expected_value)

            if not all(rounded_output == expected_vector):
                error_number += 1

                if sum(rounded_output) == 1:
                    false_positives[expected_value] += 1
            else:
                true_positives[expected_value] += 1


            error = sample_error(expected_vector, output_value) #Se calcula el error para la muestra

            sum_mse += error #Actualizar error total

        mse = sum_mse / dataset.size() #Calcular error promedio

        precision_list = []

        for key in dataset.get_labels():
            precision_list.append(round(precision(true_positives[key], false_positives[key]), 2))

        precision_string = ",".join([str(value) for value in precision_list])
        print(f'{accuracy(dataset.size(), error_number)}, {mse}, {precision_string}')


