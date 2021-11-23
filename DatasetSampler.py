import random

""" Clase que permite particionar el dataset en bloques que tienen 
    la misma proporción de todas las categorías
"""
class DatasetSampler:

    """ Constructor de la clase
        Parámetros:
            - label_list: Lista de clases que existen en el dataset
            - proportion: Proporción que debe haber de cada clase en el bloque a generar
    """
    def __init__(self, label_list, proportion):

        self.proportion = proportion
        self.class_map = dict([(key, []) for key in label_list])

    """ Mapea las clases a los índices en el dataset original que son elementos pertenecianetes a la misma
        Parámetros:
            - key: Nombre de la clase
            - index: Índice del elemento en el dataset original
    """
    def add_class_sample(self, key, index):

        self.class_map[key].append(index)

    """ Genera un bloque del tamaño de la proporción indicada respecto al dataset,  en el cual
        todas las clases están representadas en la misma proporción. Devuelve una lista con los
        elementos pertenecientes al bloque    
    """
    def make_dataset_sample(self):

        index_list = []

        for key, val in self.class_map.items():
            index_list += random.sample(val, int(self.proportion * len(val)))

        return index_list
