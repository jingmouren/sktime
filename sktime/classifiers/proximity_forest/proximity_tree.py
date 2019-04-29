from numpy.ma import floor
from scipy.stats import rv_continuous, rv_discrete, uniform, randint
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator

from classifiers.proximity_forest.randomised import Randomised
from classifiers.proximity_forest.split_score import gini
from classifiers.proximity_forest.stopping_condition import pure
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint
from distances import dtw_distance, lcss_distance, erp_distance, ddtw_distance, wddtw_distance, wdtw_distance, \
    msm_distance
from utils.transformations import tabularise

# todo docs
# todo ref paper
# todo doctests
# todo unit tests
# todo comment up!

def get_default_param_pool(self, instances):
    instance_length = 4  # todo
    max_raw_warping_window = floor((instance_length + 1) / 4)
    max_warping_window_percentage = max_raw_warping_window / instance_length
    stdp = Utilities.stdp(instances)
    param_pool = [
        {self.distance_measure_key: [dtw_distance],
         'w': uniform(0, max_warping_window_percentage)},
        {self.distance_measure_key: [ddtw_distance],
         'w': uniform(0, max_warping_window_percentage)},
        {self.distance_measure_key: [wdtw_distance],
         'g': uniform(0, 1)},
        {self.distance_measure_key: [wddtw_distance],
         'g': uniform(0, 1)},
        {self.distance_measure_key: [lcss_distance],
         'epsilon': uniform(0.2 * stdp, stdp),
         'delta': randint(low=0, high=max_raw_warping_window)},
        {self.distance_measure_key: [erp_distance],
         'g': uniform(0.2 * stdp, 0.8 * stdp),
         'band_size': randint(low=0, high=max_raw_warping_window)},
        # {self.distance_measure_key: [twe_distance],
        #  'g': uniform(0.2 * stdp, 0.8 * stdp),
        #  'band_size': randint(low=0, high=max_raw_warping_window)},
        {self.distance_measure_key: [msm_distance],
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

class ProximityTree(Randomised, BaseEstimator):
    # todo variable hinting

    stop_splitting_key = 'stop_splitting'  # todo make read only (module level func?)
    gain_key = 'gain'
    param_pool_obtainer_key = 'param_pool_obtainer'
    distance_measure_key = 'distance_measure'
    r_key = 'r'
    _exemplar_instances_key = 'exemplar_instances'
    _exemplar_class_labels_key = 'exemplar_class_labels'
    _param_perm_key = 'param_perm'
    _gain_value_key = 'gain_value'
    _distance_measure_key = 'distance_measure'
    _exemplar_bins_key = 'exemplar_bin_keys'

    default_gain = gini
    default_stop_splitting = pure
    default_param_pool_obtainer = get_default_param_pool
    default_r = 1

    def __init__(self, **params):
        self._gain = None
        self._stop_splitting = None
        self._param_pool = None
        self._param_pool_obtainer = None
        self._branches = None
        self._r = None
        self._split = None
        self._unique_class_labels = None
        super().__init__(**params)

    def predict_proba_inst(self, instance): # todo recursive, needs to be converted to iterative
        closest_exemplar_index = self._find_closest_exemplar_index(self._split[self._exemplar_instances_key],
                                                                   self._split[self._distance_measure_key],
                                                                   instance,
                                                                   self._split[self._param_perm_key],)
        # if end of tree / leaf
        if self._branches[closest_exemplar_index] is None:
            # return majority vote distribution
            distribution = np.zeros((len(self._unique_class_labels)))
            predicted_class = self._split[self._exemplar_class_labels_key][closest_exemplar_index]
            distribution[predicted_class] += 1
            return distribution
        else:
            return self._branches[closest_exemplar_index].predict_proba_inst(instance)

    def predict_proba(self, instances):
        num_instances = instances.shape[0]
        distributions = np.empty((num_instances, len(self._unique_class_labels)))
        for instance_index in range(0, num_instances):
            instance = instances.iloc[instance_index, :]
            prediction = self.predict_proba_inst(instance)
            distributions[instance_index] = prediction
        return distributions

    def predict(self, instances):
        distributions = self.predict_proba(instances)
        predictions = np.empty((distributions.shape[0]))
        for instance_index in range(0, predictions.shape[0]):
            distribution = distributions[instance_index]
            prediction = Utilities.max(distribution, self.get_rand())
            predictions[instance_index] = prediction
        return predictions

    def fit(self, instances, class_labels):
        self._unique_class_labels = np.unique(class_labels)
        binned_instances = Utilities.bin_instances_by_class(instances, class_labels)
        param_pool = self._param_pool_obtainer(instances)
        self._branch(binned_instances, param_pool) # todo err check on params + class params too
        return self

    def _branch(self, binned_instances, param_pool): # todo recursive, needs to be converted to iterative!
        self._param_pool = param_pool
        self._split = self._get_best_split(binned_instances)
        # print('best split')
        # print(self._split)
        self._branches = []
        for exemplar_bin in self._split[self._exemplar_bins_key]:
            if not self._stop_splitting(exemplar_bin):
                tree = ProximityTree(**self.get_params()) # duplicate this tree config
                self._branches.append(tree)
                tree._branch(exemplar_bin, param_pool)
            else:
                self._branches.append(None)

    def _get_rand_param_perm(self, params=None):  # list of param dicts todo split into two methods
        # example:
        # param_grid = [
        #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #   {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}], 'kernel': ['rbf']},
        #  ]
        if params is None:
            params = self._param_pool
        param_pool = self.get_rand().choice(params)
        permutation = self._pick_param_permutation(param_pool)
        return permutation

    def _get_best_split(self, binned_instances):
        split = self._pick_rand_split(binned_instances)
        # print(split[self._gain_value_key])
        if self._r > 0:
            splits = [split]
            best_gain = split[self._gain_value_key]
            for index in range(0, self._r):
                other_split = self._pick_rand_split(binned_instances)
                other_gain = other_split[self._gain_value_key]
                # format_str = '{: 1.8f}'
                # index_format_str = '{: 4d}'
                # print(index_format_str.format(index) + "    " + format_str.format(other_gain) + "    " + format_str.format(best_gain))
                if other_gain >= best_gain:
                    if other_gain > best_gain:
                        splits.clear()
                        best_gain = other_gain
                    splits.append(other_split)
            split = self.get_rand().choice(splits)
        return split

    def _pick_rand_split(self, binned_instances):
        param_perm = self._get_rand_param_perm()
        exemplar_instances, exemplar_class_labels, remaining_binned_instances = self._pick_rand_exemplars(binned_instances)
        # one split per exemplar
        exemplar_bins = []
        for exemplar in exemplar_instances:
            instances_bin = {}
            for class_label in remaining_binned_instances.keys():
                instances_bin[class_label] = []
            exemplar_bins.append(instances_bin)
        # extract distance measure
        distance_measure = param_perm[self.distance_measure_key]
        # trim distance measure to leave distance measure parameters
        del param_perm[self.distance_measure_key]
        # for each remaining instance after exemplars have been removed
        for class_label, instances_bin in remaining_binned_instances.items():
            # compare to each exemplar
            for instance in instances_bin:
                closest_exemplar_index = self._find_closest_exemplar_index(exemplar_instances, distance_measure, instance, param_perm)
                exemplar_bins[closest_exemplar_index][class_label].append(instance)
        gain = self._gain(binned_instances, *exemplar_bins)
        return {
            self._exemplar_instances_key: exemplar_instances, # todo unpack into split class with functionality for predicting
            self._exemplar_class_labels_key: exemplar_class_labels,
            self._exemplar_bins_key: exemplar_bins,
            self._param_perm_key: param_perm,
            self._distance_measure_key: distance_measure,
            self._gain_value_key: gain,
        }

    def _find_closest_exemplar_index(self, exemplar_instances, distance_measure, instance, param_perm):
        exemplar_index = 0
        exemplar_instance = exemplar_instances[exemplar_index]
        min_distance = self._find_distance(distance_measure, exemplar_instance, instance, param_perm)
        closest_exemplar_indices = [exemplar_index]
        for exemplar_index in range(1, len(exemplar_instances)):
            exemplar_instance = exemplar_instances[exemplar_index]
            distance = self._find_distance(distance_measure, exemplar_instance, instance, param_perm)
            if distance <= min_distance:
                if distance < min_distance:
                    closest_exemplar_indices.clear()
                    min_distance = distance
                closest_exemplar_indices.append(exemplar_index)
        closest_exemplar_index = self.get_rand().choice(closest_exemplar_indices)
        return closest_exemplar_index

    def _find_distance(self, distance_measure, instance_a, instance_b, param_perm):
        instance_a = tabularise(instance_a, return_array=True)
        instance_b = tabularise(instance_b, return_array=True)
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        return distance_measure(instance_a, instance_b, **param_perm)

    def _pick_param_permutation(self, param_pool):  # dict of params
        param_permutation = {}
        for param_name, param_values in param_pool.items():
            if isinstance(param_values, list):
                param_value = self.get_rand().choice(param_values)
                if isinstance(param_value, dict):
                    param_value = self._get_rand_param_perm(param_value)
            elif hasattr(param_values, 'rvs'):
                param_value = param_values.rvs(random_state=self.get_rand())
            else:
                raise Exception('unknown type')
            param_permutation[param_name] = param_value
        return param_permutation

    def _pick_rand_exemplars(self, binned_instances):
        exemplar_instances = [] # todo exemplar pick strategy
        exemplar_class_labels = []
        remaining_binned_instances = {}
        for class_label in binned_instances.keys():
            instances_bin = binned_instances[class_label]
            exemplar_index = self.get_rand().randint(0, len(instances_bin))
            exemplar = instances_bin[exemplar_index]
            remaining_instances_bin = []
            for instance_index in range(0, exemplar_index):
                instance = instances_bin[instance_index]
                remaining_instances_bin.append(instance)
            for instance_index in range(exemplar_index + 1, len(instances_bin)):
                instance = instances_bin[instance_index]
                remaining_instances_bin.append(instance)
            remaining_binned_instances[class_label] = remaining_instances_bin
            exemplar_instances.append(exemplar)
            exemplar_class_labels.append(class_label)
        return exemplar_instances, exemplar_class_labels, remaining_binned_instances

    def set_params(self, **params):
        super(ProximityTree, self).set_params(**params)
        super(ProximityTree, self)._set_param(self.gain_key, self.default_gain, params, prefix='_')
        super(ProximityTree, self)._set_param(self.r_key, self.default_r, params, prefix='_')
        super(ProximityTree, self)._set_param(self.param_pool_obtainer_key, self.default_param_pool_obtainer, params, prefix='_')
        super(ProximityTree, self)._set_param(self.stop_splitting_key, self.default_stop_splitting, params, prefix='_')

    def get_params(self, deep=False):
        return {
            ProximityTree.gain_key: self._gain,
            ProximityTree.r_key: self._r,
            ProximityTree.param_pool_obtainer_key: self._param_pool_obtainer,
            ProximityTree.stop_splitting_key: self._stop_splitting,
            **super(ProximityTree, self).get_params(),
        }

if __name__ == "__main__":

    # a = np.array([2,2,2,2], dtype=float)
    # b = np.array([3,5,1,5], dtype=float)
    # a = a.reshape((4,1))
    # b = b.reshape((4,1))
    # dist = lcss_distance(a, b, delta=1, epsilon=0.3)
    # print(dist)

    x_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    tree = ProximityTree(**{Randomised.rand_state_key: np.random.RandomState(3).get_state(), ProximityTree.r_key: 1})
    tree.fit(x_train, y_train)

