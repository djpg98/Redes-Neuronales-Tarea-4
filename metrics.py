import math
import numpy as np

def precision(true_positives, false_positives):

    try:
        return (true_positives/(true_positives+false_positives))
    except:
        return -1

def accuracy(total_data, errors):

    return ((total_data - errors)/total_data)

def sample_error(expeted_value, output_value):

    return 0.5 * np.power(expeted_value - output_value, 2).sum()