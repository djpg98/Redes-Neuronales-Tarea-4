import random

class DatasetSampler:

    def __init__(self, label_list, proportion):

        self.proportion = proportion
        self.class_map = dict([(key, []) for key in label_list])

    def add_class_sample(self, key, index):

        self.class_map[key].append(index)

    def make_dataset_sample(self):

        index_list = []

        for key, val in self.class_map.items():
            index_list += random.sample(val, int(self.proportion * len(val)))

        return index_list
