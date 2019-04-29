import numpy as np

# todo no need for class

class Utilities:
    'Utilities for common behaviour'

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def pick_from_indices(array, indices):
        picked = []
        for index in indices:
            picked.append(array[index])
        return picked

    @staticmethod
    def bests(array, comparator):
        indices = Utilities.arg_best(array, comparator)
        return Utilities.pick_from_indices(array, indices)

    @staticmethod
    def mins(array):
        indices = Utilities.arg_mins(array)
        return Utilities.pick_from_indices(array, indices)

    @staticmethod
    def maxs(array):
        indices = Utilities.arg_maxs(array)
        return Utilities.pick_from_indices(array, indices)

    @staticmethod
    def min(array, rand):
        index = Utilities.arg_min(array, rand)
        return array[index]

    @staticmethod
    def max(array, rand):
        index = Utilities.arg_max(array, rand)
        return array[index]

    @staticmethod
    def best(array, comparator, rand):
        return rand.choice(Utilities.bests(array, comparator))

    @staticmethod
    def arg_best(array, comparator, rand):
        return rand.choice(Utilities.arg_bests(array, comparator))

    @staticmethod
    def arg_min(array, rand):
        return rand.choice(Utilities.arg_mins(array))

    @staticmethod
    def arg_mins(array):
        return Utilities.arg_bests(array, Utilities.less_than)

    @staticmethod
    def arg_max(array, rand):
        return rand.choice(Utilities.arg_maxs(array))

    @staticmethod
    def arg_maxs(array):
        return Utilities.arg_bests(array, Utilities.more_than)

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

    @staticmethod
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

