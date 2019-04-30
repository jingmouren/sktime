import numpy as np

def pure(class_labels):
    unique_class_labels = np.unique(class_labels)
    return len(unique_class_labels) <= 1
