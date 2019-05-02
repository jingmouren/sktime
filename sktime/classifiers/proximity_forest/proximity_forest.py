from sklearn.preprocessing import normalize, LabelEncoder

from datasets import load_gunpoint
from utils import utilities
from utils.classifier import Classifier
from classifiers.proximity_forest.proximity_tree import ProximityTree, get_default_param_pool, gini, pure
import numpy as np

from utils.utilities import check_data

class ProximityForest(Classifier):

    def __init__(self,
                 gain_method = gini,
                 r = 5,
                 num_trees = 100,
                 rand = np.random.RandomState(),
                 is_leaf_method = pure,
                 max_depth = np.math.inf,
                 label_encoder=None,
                 param_pool = get_default_param_pool):
        super().__init__(rand=rand)
        self.gain_method = gain_method
        self.r = r
        self.label_encoder = label_encoder
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.is_leaf_method = is_leaf_method
        self.param_pool = param_pool
        # below set in fit method
        self._trees = None

    def fit(self, instances, class_labels):
        check_data(instances, class_labels)
        if self.num_trees < 1:
            raise ValueError('number of trees cannot be less than 1')
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(class_labels)
            class_labels = self.label_encoder.transform(class_labels)
        if callable(self.param_pool):
            self.param_pool = self.param_pool(instances)
        self._trees = np.empty(self.num_trees, dtype=object)
        for tree_index in np.arange(self.num_trees):
            print(tree_index)
            tree = ProximityTree(
                gain_method=self.gain_method,
                r=self.r,
                rand=self.rand,
                is_leaf_method=self.is_leaf_method,
                max_depth=self.max_depth,
                label_encoder=self.label_encoder,
                param_pool=self.param_pool
            )
            self._trees[tree_index] = tree
            tree.fit(instances, class_labels)
        return self

    def predict_proba(self, instances):
        check_data(instances)
        overall_predict_probas = np.zeros((instances.shape[0], len(self.label_encoder.classes_)))
        for tree_index in range(0, len(self._trees) - 1):
            tree = self._trees[tree_index]
            predict_probas = tree.predict_proba(instances)
            for instance_index in np.arange(predict_probas.shape[0]):
                predict_proba = predict_probas[instance_index]
                max_index = utilities.arg_max(predict_proba, self.rand)
                overall_predict_probas[instance_index][max_index] += 1
        normalize(overall_predict_probas, copy=False)
        return overall_predict_probas

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    classifier = ProximityForest(rand=np.random.RandomState(3))
    classifier.fit(x_train, y_train)

    # for tree in classifier._trees:
    #     distribution = tree.predict_proba(x_test)
    #     predictions = utilities.predict_from_distribution(distribution, classifier.rand, classifier.label_encoder)
    #     acc = utilities.accuracy(y_test, predictions)
    #     print(str(acc))

    distribution = classifier.predict_proba(x_test)
    predictions = utilities.predict_from_distribution(distribution, classifier.rand, classifier.label_encoder)
    acc = utilities.accuracy(y_test, predictions)
    print("----")
    print("acc: " + str(acc))