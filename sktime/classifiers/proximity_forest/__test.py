import numpy as np

from classifiers.proximity_forest.proximity_tree import ProximityTree
from datasets import load_gunpoint
from utils import utilities

if __name__ == "__main__":
    # a = np.array([2,2,2,2], dtype=float)
    # b = np.array([3,5,1,5], dtype=float)
    # a = a.reshape((4,1))
    # b = b.reshape((4,1))
    # dist = lcss_distance(a, b, delta=1, epsilon=0.3)
    # print(dist)

    x_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    tree = ProximityTree(rand=np.random.RandomState(0))
    tree.fit(x_train, y_train)
    distributions = tree.predict_proba(x_test)
    predictions = utilities.predict_from_distribution(distributions, tree.rand)

