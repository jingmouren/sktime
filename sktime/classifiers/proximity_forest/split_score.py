import math
import numpy as np

def gini(parent_class_labels, children_class_labels): # todo make sure this outputs 1 (max) to 0 (min) as gini is usually other way around (i.e. 0 is pure, 1 is impure (or usually 0.5))
    root_score = gini_node(parent_class_labels)
    parent_num_instances = parent_class_labels.shape[0]
    children_score_sum = 0
    for index in np.arange(len(children_class_labels)):
        child_class_labels = children_class_labels[index]
        child_score = gini_node(child_class_labels)
        child_size = len(child_class_labels)
        child_score *= (child_size / parent_num_instances)
        children_score_sum += child_score
    score = root_score - children_score_sum
    return score

def gini_node(class_labels):
    num_instances = class_labels.shape[0]
    score = 1
    if num_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(class_labels, return_counts=True)
        for index in np.arange(len(unique_class_labels)):
            class_count = class_counts[index]
            proportion = class_count / num_instances
            score -= math.pow(proportion, 2)
    return score

def information_gain(parent_class_labels, children_class_labels):
    raise Exception('not implemented yet')

def chi_squared(parent_class_labels, children_class_labels):
    raise Exception('not implemented yet')