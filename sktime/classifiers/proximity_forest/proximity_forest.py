from sklearn.preprocessing import normalize

from utils.classifier import Classifier
from classifiers.proximity_forest.proximity_tree import ProximityTree, get_default_param_pool
from classifiers.proximity_forest.split_score import gini
from classifiers.proximity_forest.stopping_condition import pure
from utils.utilities import Utilities
import numpy as np

# todo checks on num tree, r, + prox tree fields
# todo gain method
# todo stopping condition (for splits)
# todo duck typing
# todo replace lists with np arrays where poss + on prox tree
# todo tree depth? basically stopping criteria
# todo pycharm presets, e.g. auto optimise imports

class ProximityForest(Classifier):

    def __init__(self,
                 gain_method = gini,
                 r = 1,
                 num_trees = 100,
                 rand = np.random.RandomState(),
                 stop_splitting = pure,
                 param_pool_obtainer = get_default_param_pool):
        super(Classifier, self).__init__()
        self.gain_method = gain_method
        self.r = r
        self._rand = rand
        self.num_trees = num_trees
        self.stop_splitting = stop_splitting
        self.param_pool_obtainer = param_pool_obtainer
        # below set in fit method
        self._unique_class_labels = None

    def fit(self, instances, class_labels):
        self._unique_class_labels = np.unique(class_labels)
        self._trees = np.empty((self.num_trees))
        for tree_index in range(0, self.num_trees - 1):
            tree = ProximityTree(**self.get_params())
            self._trees[tree_index] = tree
            tree.fit(instances, class_labels)
        return self

    def predict_proba(self, instances):
        overall_predict_probas = np.zeros((instances.shape[0], len(self._unique_class_labels)))
        for tree_index in range(0, len(self._trees) - 1):
            tree = self._trees[tree_index]
            predict_probas = tree.predict_proba(instances)
            for instance_index in range(0, predict_probas.shape[0]):
                predict_proba = predict_probas[instance_index]
                max_index = Utilities.arg_max(predict_proba, self._rand)
                overall_predict_probas[instance_index][max_index] = 1
        for instance_index in range(0, overall_predict_probas.shape[0]):
            predict_proba = overall_predict_probas[instance_index]
            normalize(predict_proba, copy=False)
        return overall_predict_probas


# if __name__ == "__main__":
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