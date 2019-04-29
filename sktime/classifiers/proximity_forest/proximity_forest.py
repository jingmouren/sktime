from sklearn.base import BaseEstimator

from classifiers.proximity_forest.proximity_tree import ProximityTree
from classifiers.proximity_forest.randomised import Randomised
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint
import numpy as np

# todo checks on num tree, r, + prox tree fields
# todo gain method
# todo stopping condition (for splits)
# todo replace lists with np arrays where poss + on prox tree

class ProximityForest(Randomised, BaseEstimator):

    @staticmethod
    def get_num_trees_key():
        return 'num_trees'

    @staticmethod
    def _get_trees_key():
        return 'trees'

    def __init__(self, **params):
        self.num_trees = None
        self._trees = None
        self._unique_class_labels = None
        super(ProximityForest, self).__init__(**params)

    def get_params(self):
        return {
            self.get_num_trees_key(): self.num_trees
        }

    def set_params(self, **params):
        raise NotImplementedError()

    def fit(self, instances, class_labels):
        self._unique_class_labels = np.unique(class_labels)
        self._trees = np.empty((self.num_trees))
        for tree_index in range(0, self.num_trees - 1):
            tree = ProximityTree(**self.get_params())
            self._trees[tree_index] = tree
            tree.fit(instances, class_labels)

    def predict(self, instances):
        pass

    def predict_proba(self, instances):
        for instance_index in len


        votes = np.zeros(len(self._unique_class_labels))
        for tree_index in range(0, len(self._trees)):
            tree = self._trees[tree_index]
            distributions = tree.predict_proba(instances)
            for distribution in distributions:
                prediction = Utilities.max(distribution, self.get_rand())
                votes[tre[prediction] += 1


if __name__ == "__main__":
    # a = np.array([2,2,2,2], dtype=float)
    # b = np.array([3,5,1,5], dtype=float)
    # a = a.reshape((4,1))
    # b = b.reshape((4,1))
    # dist = lcss_distance(a, b, delta=1, epsilon=0.3)
    # print(dist)

    # x_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    # x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    # tree = ProximityTree(**{Randomised.rand_state_key: 0})
    # tree.fit(x_train, y_train)