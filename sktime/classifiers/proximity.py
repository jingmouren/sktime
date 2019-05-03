
# Proximity Forest: An effective and scalable distance-based classifier for time series
#
# author: George Oastler (linkedin.com/goastler)
#
# paper link: https://arxiv.org/abs/1808.10594
# bibtex reference:
    # @article{DBLP:journals/corr/abs-1808-10594,
    #   author    = {Benjamin Lucas and
    #                Ahmed Shifaz and
    #                Charlotte Pelletier and
    #                Lachlan O'Neill and
    #                Nayyar A. Zaidi and
    #                Bart Goethals and
    #                Fran{\c{c}}ois Petitjean and
    #                Geoffrey I. Webb},
    #   title     = {Proximity Forest: An effective and scalable distance-based classifier
    #                for time series},
    #   journal   = {CoRR},
    #   volume    = {abs/1808.10594},
    #   year      = {2018},
    #   url       = {http://arxiv.org/abs/1808.10594},
    #   archivePrefix = {arXiv},
    #   eprint    = {1808.10594},
    #   timestamp = {Mon, 03 Sep 2018 13:36:40 +0200},
    #   biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-10594},
    #   bibsource = {dblp computer science bibliography, https://dblp.org}
    # }
#
# todo docs
# todo doctests?
# todo unit tests
# todo comment up!
# todo score
# todo param docs
# todo use generators in indexed for loops
# todo parallel for proxtree, proxfor and proxstump? (might not work that last one!)
from numpy.ma import floor
from pandas import DataFrame, Series
from scipy.stats import uniform, randint
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import normalize, LabelEncoder
from utils import utilities
from utils.classifier import Classifier
from datasets import load_gunpoint
from distances import dtw_distance, lcss_distance, erp_distance, ddtw_distance, wddtw_distance, wdtw_distance, \
    msm_distance
from utils.transformations import tabularise
from utils.utilities import check_data

# test whether a set of class labels are pure (i.e. all the same)
def pure(class_labels):
    # get unique class labels
    unique_class_labels = np.unique(class_labels)
    # if more than 1 unique then not pure
    return len(unique_class_labels) <= 1

# get gini score of a split, i.e. the gain from parent to children
def gini(parent_class_labels, children_class_labels):
    # find gini for parent node
    parent_score = gini_node(parent_class_labels)
    # find number of instances overall
    parent_num_instances = parent_class_labels.shape[0]
    # sum the children's gini scores
    children_score_sum = 0
    for index in range(0,len(children_class_labels)):
        child_class_labels = children_class_labels[index]
        # find gini score for this child
        child_score = gini_node(child_class_labels)
        # weight score by proportion of instances at child compared to parent
        child_size = len(child_class_labels)
        child_score *= (child_size / parent_num_instances)
        # add to cumulative sum
        children_score_sum += child_score
    # gini outputs relative improvement
    score = parent_score - children_score_sum
    return score

# get gini score at a specific node
def gini_node(class_labels):
    # get number instances at node
    num_instances = class_labels.shape[0]
    score = 1
    if num_instances > 0:
        # count each class
        unique_class_labels, class_counts = np.unique(class_labels, return_counts=True)
        # subtract class entropy from current score for each class
        for index in range(0,len(unique_class_labels)):
            class_count = class_counts[index]
            proportion = class_count / num_instances
            sq_proportion = np.math.pow(proportion, 2)
            score -= sq_proportion
    # double score as gini is between 0 and 0.5, we need 0 and 1
    score *= 2
    return score

# todo
def information_gain(parent_class_labels, children_class_labels):
    raise Exception('not implemented yet')

# todo
def chi_squared(parent_class_labels, children_class_labels):
    raise Exception('not implemented yet')

# pick one random exemplar instance per class
def pick_rand_exemplars(instances, class_labels, rand):
    # find unique class labels
    unique_class_labels = np.unique(class_labels)
    num_unique_class_labels = len(unique_class_labels)
    chosen_instances = np.empty(num_unique_class_labels, dtype=object)
    chosen_class_labels = np.empty(num_unique_class_labels, dtype=int)
    chosen_indices = np.empty(num_unique_class_labels, dtype=int)
    # for each class randomly choose and instance
    for class_label_index in range(0,num_unique_class_labels):
        class_label = unique_class_labels[class_label_index]
        # filter class labels for desired class and get indices
        indices = np.argwhere(class_labels == class_label)
        # flatten numpy output
        indices = np.ravel(indices)
        # random choice
        index = rand.choice(indices)
        # record exemplar instance and class label
        instance = instances.iloc[index, :]
        chosen_instances[class_label_index] = instance
        chosen_class_labels[class_label_index] = class_label
        chosen_indices[class_label_index] = index
    # remove exemplar class labels from dataset - note this returns a copy, not inplace!
    class_labels = np.delete(class_labels, chosen_indices)
    # remove exemplar instances from dataset - note this returns a copy, not inplace!
    instances = instances.drop(instances.index[chosen_indices])
    return chosen_instances, chosen_class_labels, instances, class_labels

# default parameter pool of distance measures and corresponding parameters derived from a dataset
def get_default_param_pool(instances):
    # find dataset properties
    instance_length = utilities.max_instance_length(
        instances)  # todo should this use the max instance length for unequal length dataset instances?
    max_raw_warping_window = floor((instance_length + 1) / 4)
    max_warping_window_percentage = max_raw_warping_window / instance_length
    stdp = utilities.stdp(instances)
    # setup param pool dictionary array (same structure as sklearn's GridSearchCV params!)
    param_pool = [
        {ProximityStump.get_distance_measure_key(): [dtw_distance],
         'w': uniform(0, max_warping_window_percentage)},
        {ProximityStump.get_distance_measure_key(): [ddtw_distance],
         'w': uniform(0, max_warping_window_percentage)},
        {ProximityStump.get_distance_measure_key(): [wdtw_distance],
         'g': uniform(0, 1)},
        {ProximityStump.get_distance_measure_key(): [wddtw_distance],
         'g': uniform(0, 1)},
        {ProximityStump.get_distance_measure_key(): [lcss_distance],
         'epsilon': uniform(0.2 * stdp, stdp),
         'delta': randint(low=0, high=max_raw_warping_window)},
        {ProximityStump.get_distance_measure_key(): [erp_distance],
         'g': uniform(0.2 * stdp, 0.8 * stdp),
         'band_size': randint(low=0, high=max_raw_warping_window)},
        # {Split.get_distance_measure_key(): [twe_distance],
        #  'g': uniform(0.2 * stdp, 0.8 * stdp),
        #  'band_size': randint(low=0, high=max_raw_warping_window)},
        {ProximityStump.get_distance_measure_key(): [msm_distance],
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

# proximity tree classifier of depth 1 - in other words, a k=1 nearest neighbour classifier with neighbourhood limited
# to x exemplar instances
class ProximityStump(Classifier):

    def __init__(self,
                 pick_exemplars_method = None,
                 param_perm = None,
                 gain_method = None,
                 label_encoder = None,
                 rand = np.random.RandomState):
        super().__init__(rand=rand)
        self.param_perm = param_perm
        self.gain_method = gain_method
        self.pick_exemplars_method = pick_exemplars_method
        # vars set in the fit method
        self._distances = None
        self.exemplar_instances = None
        self.exemplar_class_labels = None
        self.remaining_instances = None
        self.remaining_class_labels = None
        self.branch_instances = None
        self.branch_class_labels = None
        self.distance_measure_param_perm = None
        self.distance_measure = None
        self.gain = None
        self.label_encoder = label_encoder
        self.classes_ = None

    @staticmethod
    def get_distance_measure_key():
        return 'dm'

    def fit(self, instances, class_labels):
        check_data(instances, class_labels)
        if callable(self.param_perm):
            self.param_perm = self.param_perm(instances)
        if not isinstance(self.param_perm, dict):
            raise ValueError("parameter permutation must be a dict or callable to obtain dict")
        if not callable(self.gain_method):
            raise ValueError("gain method must be callable")
        if not callable(self.pick_exemplars_method):
            raise ValueError("gain method must be callable")
        if not isinstance(self.rand, np.random.RandomState):
            raise ValueError('rand not set to a random state')
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(class_labels)
            class_labels = self.label_encoder.transform(class_labels)
        if self.distance_measure is None:
            key = self.get_distance_measure_key()
            self.distance_measure = self.param_perm[key]
            self.distance_measure_param_perm = self.param_perm.copy()
            del self.distance_measure_param_perm[key]
        self.classes_ = self.label_encoder.classes_
        self.exemplar_instances, self.exemplar_class_labels, self.remaining_instances, self.remaining_class_labels = \
            self.pick_exemplars_method(instances, class_labels, self.rand)
        distances = self.exemplar_distances(self.remaining_instances)
        num_exemplars = len(self.exemplar_instances)
        self.branch_class_labels = np.empty(num_exemplars, dtype=object)
        self.branch_instances = np.empty(num_exemplars, dtype=object)
        for index in range(0,num_exemplars):
            self.branch_instances[index] = []
            self.branch_class_labels[index] = []
        num_instances = self.remaining_instances.shape[0]
        for instance_index in range(0,num_instances):
            exemplar_distances = distances[instance_index]
            instance = self.remaining_instances.iloc[instance_index, :]
            class_label = self.remaining_class_labels[instance_index]
            closest_exemplar_index = utilities.arg_min(exemplar_distances, self.rand)
            self.branch_instances[closest_exemplar_index].append(instance)
            self.branch_class_labels[closest_exemplar_index].append(class_label)
        for index in range(0,self.branch_class_labels.shape[0]):
            self.branch_class_labels[index] = np.array(self.branch_class_labels[index])
            self.branch_instances[index] = DataFrame(self.branch_instances[index])
        self.gain = self.gain_method(class_labels, self.branch_class_labels)
        return self

    def exemplar_distances(self, instances):
        check_data(instances)
        num_instances = instances.shape[0]
        num_exemplars = len(self.exemplar_instances)
        distances = np.zeros((num_instances, num_exemplars))
        for instance_index in range(0,num_instances):
            instance = instances.iloc[instance_index, :]
            self.exemplar_distance_inst(instance, distances[instance_index])
        return distances

    def exemplar_distance_inst(self, instance, distances=None):
        if not isinstance(instance, Series):
            raise ValueError("instance not a panda series")
        num_exemplars = len(self.exemplar_instances)
        if distances is None:
            distances = np.zeros(num_exemplars)
        for exemplar_index in range(0,num_exemplars):
            exemplar = self.exemplar_instances[exemplar_index]
            distance = self._find_distance(exemplar, instance)
            distances[exemplar_index] = distance
        return distances

    def predict_proba(self, instances):
        check_data(instances)
        num_instances = instances.shape[0]
        num_exemplars = len(self.exemplar_instances)
        num_unique_class_labels = len(self.label_encoder.classes_)
        distributions = np.empty((num_instances, num_unique_class_labels))
        distances = self.exemplar_distances(instances)
        for instance_index in range(0,num_instances):
            distribution = np.zeros(num_unique_class_labels)
            distributions[instance_index] = distribution
            for exemplar_index in range(0,num_exemplars):
                distance = distances[instance_index][exemplar_index]
                exemplar_class_label = self.exemplar_class_labels[exemplar_index]
                distribution[exemplar_class_label - 1] -= distance
            max_distance = -np.min(distribution)
            for exemplar_index in range(0, num_exemplars - 1):
                distribution[exemplar_index] += max_distance
        normalize(distributions, copy=False)
        return distributions

    def _find_distance(self, instance_a, instance_b):
        if not isinstance(instance_a, Series):
            raise ValueError("instance not a panda series")
        if not isinstance(instance_b, Series):
            raise ValueError("instance not a panda series")
        instance_a = tabularise(instance_a, return_array=True)
        instance_b = tabularise(instance_b, return_array=True)
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        return self.distance_measure(instance_a, instance_b, **self.distance_measure_param_perm)


class ProximityTree(Classifier): # todd rename split to stump

    def __init__(self,
                 gain_method=gini,
                 r=5,
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
        self.classes_ = None

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
            for branch_index in range(0,num_branches):
                exemplar_class_labels = self._split.branch_class_labels[branch_index]
                if not self.is_leaf_method(exemplar_class_labels):
                    tree = clone(self, safe=True)
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
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(class_labels)
            class_labels = self.label_encoder.transform(class_labels)
        self.classes_ = self.label_encoder.classes_
        tree_stack = [self]
        instances_stack = [instances]
        class_labels_stack = [class_labels]
        while tree_stack:
            tree = tree_stack.pop()
            instances = instances_stack.pop()
            class_labels = class_labels_stack.pop()
            tree._branch(instances, class_labels)
            for branch_index in range(0,len(tree._branches)):
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
        for index in range(0,self.r):
            split = self._pick_rand_split(instances, class_labels)
            splits[index] = split
        best_split = utilities.best(splits, lambda a, b: b.gain - a.gain, self.rand)
        return best_split

    def _pick_rand_split(self, instances, class_labels):
        param_perm = self._get_rand_param_perm()
        split = ProximityStump(pick_exemplars_method=self.pick_exemplars_method,
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
        self.classes_ = None

    def _generate_tree(self, instances, class_labels):
        tree = ProximityTree(
            gain_method=self.gain_method,
            r=self.r,
            rand=self.rand,
            is_leaf_method=self.is_leaf_method,
            max_depth=self.max_depth,
            label_encoder=self.label_encoder,
            param_pool=self.param_pool
        )
        tree.fit(instances, class_labels)
        return tree

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
        self.classes_ = self.label_encoder.classes_
        self._trees = np.empty(self.num_trees, dtype=object)
        for tree_index in range(0, self.num_trees):
            print("tree index: " + str(tree_index))
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
        # overall_predict_probas = np.zeros((instances.shape[0], len(self.label_encoder.classes_)))
        overall_predict_probas = np.zeros((instances.shape[0], len(self.label_encoder.classes_)))
        for tree in self._trees:
            predict_probas = tree.predict_proba(instances)
            overall_predict_probas = np.add(overall_predict_probas, predict_probas)
        normalize(overall_predict_probas, copy=False, norm='l1')
        return overall_predict_probas

# todo debug option to do helpful printing