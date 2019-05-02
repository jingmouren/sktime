from numpy.ma import floor
from scipy.stats import uniform, randint
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from utils import utilities
from utils.classifier import Classifier
from classifiers.proximity_forest.split import Split
from datasets import load_gunpoint
from distances import dtw_distance, lcss_distance, erp_distance, ddtw_distance, wddtw_distance, wdtw_distance, \
    msm_distance

# todo docs
# todo ref paper
# todo doctests
# todo unit tests
# todo comment up!
# todo mixin
# todo score
# todo return class label from predict, not index? Use label thing tony found - labels currently strings D:
# todo parallel for proxtree and proxfor

from utils.utilities import check_data


def pure(class_labels):
    unique_class_labels = np.unique(class_labels)
    return len(unique_class_labels) <= 1


def gini(parent_class_labels,
         children_class_labels):  # todo make sure this outputs 1 (max) to 0 (min) as gini is usually other way around (i.e. 0 is pure, 1 is impure (or usually 0.5))
    root_score = _gini_node(parent_class_labels)
    parent_num_instances = parent_class_labels.shape[0]
    children_score_sum = 0
    for index in np.arange(len(children_class_labels)):
        child_class_labels = children_class_labels[index]
        child_score = _gini_node(child_class_labels)
        child_size = len(child_class_labels)
        child_score *= (child_size / parent_num_instances)
        children_score_sum += child_score
    score = root_score - children_score_sum
    return score


def _gini_node(class_labels):
    num_instances = class_labels.shape[0]
    score = 1
    if num_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(class_labels, return_counts=True)
        for index in np.arange(len(unique_class_labels)):
            class_count = class_counts[index]
            proportion = class_count / num_instances
            score -= np.math.pow(proportion, 2)
    return score


def information_gain(parent_class_labels, children_class_labels):
    raise Exception('not implemented yet')


def chi_squared(parent_class_labels, children_class_labels):
    raise Exception('not implemented yet')


def pick_rand_exemplars(instances, class_labels, rand):
    unique_class_labels = np.unique(class_labels)
    num_unique_class_labels = len(unique_class_labels)
    chosen_instances = np.empty(num_unique_class_labels, dtype=object)
    chosen_class_labels = np.empty(num_unique_class_labels, dtype=int)
    chosen_indices = np.empty(num_unique_class_labels, dtype=int)
    for class_label_index in np.arange(num_unique_class_labels):
        class_label = unique_class_labels[class_label_index]
        indices = np.argwhere(class_labels == class_label)
        indices = np.ravel(indices)
        index = rand.choice(indices)
        instance = instances.iloc[index, :]
        chosen_instances[class_label_index] = instance
        chosen_class_labels[class_label_index] = class_label
        chosen_indices[class_label_index] = index
    class_labels = np.delete(class_labels, chosen_indices)
    instances = instances.drop(instances.index[chosen_indices])
    return chosen_instances, chosen_class_labels, instances, class_labels


def get_default_param_pool(instances):
    instance_length = utilities.max_instance_length(
        instances)  # todo should this use the max instance length for unequal length dataset instances?
    max_raw_warping_window = floor((instance_length + 1) / 4)
    max_warping_window_percentage = max_raw_warping_window / instance_length
    stdp = utilities.stdp(instances)
    param_pool = [
        {Split.get_distance_measure_key(): [dtw_distance],
         'w': uniform(0, max_warping_window_percentage)},
        {Split.get_distance_measure_key(): [ddtw_distance],
         'w': uniform(0, max_warping_window_percentage)},
        {Split.get_distance_measure_key(): [wdtw_distance],
         'g': uniform(0, 1)},
        {Split.get_distance_measure_key(): [wddtw_distance],
         'g': uniform(0, 1)},
        {Split.get_distance_measure_key(): [lcss_distance],
         'epsilon': uniform(0.2 * stdp, stdp),
         'delta': randint(low=0, high=max_raw_warping_window)},
        {Split.get_distance_measure_key(): [erp_distance],
         'g': uniform(0.2 * stdp, 0.8 * stdp),
         'band_size': randint(low=0, high=max_raw_warping_window)},
        # {Split.get_distance_measure_key(): [twe_distance],
        #  'g': uniform(0.2 * stdp, 0.8 * stdp),
        #  'band_size': randint(low=0, high=max_raw_warping_window)},
        {Split.get_distance_measure_key(): [msm_distance],
         'c': [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375, 0.0475, 0.05125,
               0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125, 0.085, 0.08875, 0.0925, 0.09625,
               0.1, 0.136, 0.172, 0.208,
               0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748,
               0.784, 0.82, 0.856,
               0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68,
               6.04, 6.4, 6.76, 7.12,
               7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
               49.6, 53.2, 56.8, 60.4,
               64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100]},
    ]
    return param_pool


class ProximityTree(Classifier):

    def __init__(self,
                 gain_method=gini,
                 r=1,
                 max_depth=np.math.inf,
                 rand=np.random.RandomState(),
                 is_leaf_method=pure,
                 level=0,
                 label_encoder=None,
                 pick_exemplars_method=pick_rand_exemplars,
                 param_pool=get_default_param_pool):
        super().__init__(rand=rand)
        self.gain_method = gain_method
        self.r = r
        self.max_depth = max_depth
        self.level = level
        self.label_encoder = label_encoder
        self.pick_exemplars_method = pick_exemplars_method
        self.is_leaf_method = is_leaf_method
        self.param_pool = param_pool
        # vars set in the fit method
        self._branches = None
        self._split = None

    def predict_proba(self, instances):
        check_data(instances)
        num_instances = instances.shape[0]
        distributions = np.empty((num_instances, len(self.label_encoder.classes_)))
        for instance_index in range(0, num_instances):
            instance = instances.iloc[instance_index, :]
            previous_tree = None
            tree = self
            closest_exemplar_index = -1
            while tree:
                distances = tree._split.exemplar_distance_inst(instance)
                closest_exemplar_index = utilities.arg_min(distances, tree.rand)
                previous_tree = tree
                tree = tree._branches[closest_exemplar_index]
            tree = previous_tree
            prediction = np.zeros(len(self.label_encoder.classes_))
            closest_exemplar_class_label = tree._split.exemplar_class_labels[closest_exemplar_index]
            prediction[closest_exemplar_class_label] += 1
            distributions[instance_index] = prediction
        return distributions

    def _branch(self, instances, class_labels):
        self._split = self._get_best_split(instances, class_labels)
        num_branches = len(self._split.branch_instances)
        self._branches = np.empty(num_branches, dtype=object)
        if self.level < self.max_depth:
            for branch_index in np.arange(num_branches):
                exemplar_class_labels = self._split.branch_class_labels[branch_index]
                if not self.is_leaf_method(exemplar_class_labels):
                    tree = clone(self)
                    tree.label_encoder = self.label_encoder
                    tree.depth = self.level + 1
                    self._branches[branch_index] = tree
                else:
                    self._branches[branch_index] = None

    def fit(self, instances, class_labels):
        check_data(instances, class_labels)
        if self.level is None or self.level < 0:
            raise ValueError('depth cannot be less than 0 or None')
        if self.max_depth < 0:
            raise ValueError('max depth cannot be less than 0')
        if self.r < 1:
            raise ValueError('r cannot be less than 1')
        if not callable(self.gain_method):
            raise RuntimeError('gain method not callable')
        if not callable(self.pick_exemplars_method):
            raise RuntimeError('pick exemplars method not callable')
        if not callable(self.is_leaf_method):
            raise RuntimeError('is leaf method not callable')
        if callable(self.param_pool):
            self.param_pool = self.param_pool(instances)
        if not isinstance(self.rand, np.random.RandomState):
            raise ValueError('rand not set to a random state')
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(class_labels)
        class_labels = self.label_encoder.transform(class_labels)
        tree_stack = [self]
        instances_stack = [instances]
        class_labels_stack = [class_labels]
        while tree_stack:
            tree = tree_stack.pop()
            instances = instances_stack.pop()
            class_labels = class_labels_stack.pop()
            tree._branch(instances, class_labels)
            for branch_index in np.arange(len(tree._branches)):
                sub_tree = tree._branches[branch_index]
                if sub_tree is not None:
                    tree_stack.insert(0, sub_tree)
                    instances = tree._split.branch_instances[branch_index]
                    class_labels = tree._split.branch_class_labels[branch_index]
                    instances_stack.insert(0, instances)
                    class_labels_stack.insert(0, class_labels)
        return self

    def _get_rand_param_perm(self, params=None):
        # example:
        # param_grid = [
        #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #   {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}], 'kernel': ['rbf']},
        #  ]
        if params is None:
            params = self.param_pool
        param_pool = self.rand.choice(params)
        permutation = self._pick_param_permutation(param_pool)
        return permutation

    def _get_best_split(self, instances, class_labels):
        splits = np.empty(self.r, dtype=object)
        for index in np.arange(self.r):
            split = self._pick_rand_split(instances, class_labels)
            splits[index] = split
        best_split = utilities.best(splits, lambda a, b: b.gain - a.gain, self.rand)
        return best_split

    def _pick_rand_split(self, instances, class_labels):
        param_perm = self._get_rand_param_perm()
        split = Split(pick_exemplars_method=self.pick_exemplars_method,
                      rand=self.rand,
                      gain_method=self.gain_method,
                      param_perm=param_perm)
        split.fit(instances, class_labels)
        return split

    def _pick_param_permutation(self, param_pool):  # dict of params
        param_permutation = {}
        for param_name, param_values in param_pool.items():
            if isinstance(param_values, list):
                param_value = self.rand.choice(param_values)
                if isinstance(param_value, dict):
                    param_value = self._get_rand_param_perm(param_value)
            elif hasattr(param_values, 'rvs'):
                param_value = param_values.rvs(random_state=self.rand)
            else:
                raise Exception('unknown type')
            param_permutation[param_name] = param_value
        return param_permutation


if __name__ == "__main__":
    # a = np.array([2,2,2,2], dtype=float)
    # b = np.array([3,5,1,5], dtype=float)
    # a = a.reshape((4,1))
    # b = b.reshape((4,1))
    # dist = lcss_distance(a, b, delta=1, epsilon=0.3)
    # print(dist)

    x_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    tree = ProximityTree(rand=np.random.RandomState(3), r=1)
    tree.fit(x_train, y_train)
    distribution = tree.predict_proba(x_test)
    predictions = utilities.predict_from_distribution(distribution, tree.rand, tree.label_encoder)
    print(predictions)