from sklearn.preprocessing import normalize

from utils import utilities
from utils.classifier import Classifier
from classifiers.proximity_forest.proximity_tree import ProximityTree, get_default_param_pool, gini, pure
import numpy as np

from utils.utilities import check_data

class ProximityForest(Classifier):

    def __init__(self,
                 gain_method = gini,
                 r = 1,
                 num_trees = 100,
                 rand = np.random.RandomState(),
                 is_leaf_method = pure,
                 max_tree_depth = np.math.inf,
                 param_pool_obtainer = get_default_param_pool):
        super().__init__(rand=rand)
        self.gain_method = gain_method
        self.r = r
        self.max_tree_depth = max_tree_depth
        self.num_trees = num_trees
        self.is_leaf_method = is_leaf_method
        self.param_pool_obtainer = param_pool_obtainer
        # below set in fit method
        self._trees = None
        self._unique_class_labels = None

    def fit(self, instances, class_labels):
        check_data(instances, class_labels)
        if self.num_trees < 1:
            raise ValueError('number of trees cannot be less than 1')
        self._unique_class_labels = np.unique(class_labels)
        self._trees = np.empty(self.num_trees)
        for tree_index in range(0, self.num_trees - 1):
            tree = ProximityTree(**self.get_params(), level=0)
            self._trees[tree_index] = tree
            tree.fit(instances, class_labels)
        return self

    def predict_proba(self, instances):
        check_data(instances)
        overall_predict_probas = np.zeros((instances.shape[0], len(self._unique_class_labels)))
        for tree_index in range(0, len(self._trees) - 1):
            tree = self._trees[tree_index]
            predict_probas = tree.predict_proba(instances)
            for instance_index in range(0, predict_probas.shape[0]):
                predict_proba = predict_probas[instance_index]
                max_index = utilities.arg_max(predict_proba, self.rand)
                overall_predict_probas[instance_index][max_index] = 1
        for instance_index in range(0, overall_predict_probas.shape[0]):
            predict_proba = overall_predict_probas[instance_index]
            normalize(predict_proba, copy=False)
        return overall_predict_probas