import numpy as np
from pandas import DataFrame

def predict_from_distribution(distributions, rand):
    predictions = np.empty((distributions.shape[0]))
    for instance_index in range(0, predictions.shape[0]):
        distribution = distributions[instance_index]
        prediction = max(distribution, rand)
        predictions[instance_index] = prediction
    return predictions

def check_data(instances, class_labels=None):
    if not isinstance(instances, DataFrame):
        raise ValueError("instances not in panda dataframe")
    if class_labels is not None:
        # todo check size
        raise NotImplementedError()

def stdp(instances):
    sum = 0
    sum_sq = 0
    num_instances = instances.shape[0]
    num_dimensions = instances.shape[1]
    num_values = num_instances * num_dimensions
    for instance_index in range(0, num_instances):
        for dimension_index in range(0, num_dimensions):
            instance = instances.iloc[instance_index, dimension_index]
            for value in instance:
                sum += value
                sum_sq += (value ** 2)  # todo missing values NaN messes this up!
    mean = sum / num_values
    stdp = np.math.sqrt(sum_sq / num_values - mean ** 2)
    return stdp


def arg_bests(array, comparator):
    indices = [0]
    best = array[0]
    for index in range(1, len(array)):
        value = array[index]
        comparison_result = comparator(value, best)
        if comparison_result >= 0:
            if comparison_result > 0:
                indices = []
                best = value
            indices.append(index)
    return indices


def pick_from_indices(array, indices):
    picked = []
    for index in indices:
        picked.append(array[index])
    return picked


def bests(array, comparator):
    indices = arg_best(array, comparator)
    return pick_from_indices(array, indices)


def mins(array):
    indices = arg_mins(array)
    return pick_from_indices(array, indices)


def maxs(array):
    indices = arg_maxs(array)
    return pick_from_indices(array, indices)


def min(array, rand):
    index = arg_min(array, rand)
    return array[index]


def max(array, rand):
    index = arg_max(array, rand)
    return array[index]


def best(array, comparator, rand):
    return rand.choice(bests(array, comparator))


def arg_best(array, comparator, rand):
    return rand.choice(arg_bests(array, comparator))


def arg_min(array, rand):
    return rand.choice(arg_mins(array))


def arg_mins(array):
    return arg_bests(array, less_than)


def arg_max(array, rand):
    return rand.choice(arg_maxs(array))


def arg_maxs(array):
    return arg_bests(array, more_than)


def less_than(a, b):
    if b < a:
        return -1
    elif b > a:
        return 1
    else:
        return 0


def more_than(a, b):
    if b < a:
        return 1
    elif b > a:
        return -1
    else:
        return 0


def bin_instances_by_class(instances, class_labels):
    bins = {}
    for class_label in np.unique(class_labels):
        bins[class_label] = []
    num_instances = instances.shape[0]
    for instance_index in range(0, num_instances):
        instance = instances.iloc[instance_index, :]
        class_label = class_labels[instance_index]
        instances_bin = bins[class_label]
        instances_bin.append(instance)
    return bins


def max_instance_length(instances):
    num_instances = instances.shape[0]
    max = -1
    for instance_index in range(0, num_instances):
        for dim_index in range(0, instances.shape[1]):
            instance = instances.iloc[instance_index, dim_index]
            if len(instance) > max:
                max = len(instance)
    return max
