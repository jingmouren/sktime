from numpy.ma import floor
from scipy.stats import rv_continuous, rv_discrete, uniform, randint
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from classifiers.proximity_forest.randomised import Randomised
from classifiers.proximity_forest.split_score import gini
from classifiers.proximity_forest.stopping_condition import pure
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint
from distances import dtw_distance, lcss_distance, erp_distance, ddtw_distance, wddtw_distance, wdtw_distance, \
    msm_distance


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

class ProximityTree(Randomised, BaseEstimator, ClassifierMixin):
    # todo variable hinting

    stop_splitting_key = 'stop_splitting'  # todo make read only (module level func?)
    gain_key = 'gain'
    exemplar_class_key = 'exemplar_class'
    exemplar_instance_key = 'exemplar_instance'
    param_pool_obtainer_key = 'param_pool_obtainer'
    distance_measure_key = 'distance_measure'
    r_key = 'r'
    _exemplars_key = 'exemplars'
    _param_perm_key = 'param_perm'
    _gain_value_key = 'gain_value'

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
        super().__init__(**params)

    def _predict_proba_inst(self, instance):
        pass

    def predict_proba(self, instances):
        # todo unpack panda
        pass

    def fit(self, instances, class_labels):
        binned_instances = Utilities.bin_instances_by_class(instances, class_labels)
        self._param_pool = self._param_pool_obtainer(instances)
        self._branches = []
        self._branch(binned_instances)

    def _branch(self, binned_instances):
        self._split = self._get_best_split(binned_instances)
        split = self._split
        if not self._stop_splitting(**split):
            for class_label, instances_bin in binned_instances.items():
                tree = ProximityTree(**self.get_params())
                self._branches.append(tree)
                tree._branch(instances_bin)

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


    def _pick_rand_split(self, binned_instances):
        param_perm = self._get_rand_param_perm()
        exemplars, remaining_binned_instances = self._pick_rand_exemplars(binned_instances)
        # one split per exemplar
        splits = []
        for exemplar in exemplars:
            exemplar_instance_bin = {}
            for class_label in remaining_binned_instances.keys():
                exemplar_instance_bin[class_label] = []
            splits.append(exemplar_instance_bin)
        # extract distance measure
        distance_measure = param_perm[self.distance_measure_key]
        # trim distance measure to leave distance measure parameters
        del param_perm[self.distance_measure_key]
        # for each remaining instance after exemplars have been removed
        for class_label, instances_bin in remaining_binned_instances.items():
            # compare to each exemplar
            for instance in instances_bin:
                min_distance = None
                closest_exemplar_indices = []
                for exemplar_index in range(0, len(exemplars)):
                    exemplar = exemplars[exemplar_index]
                    distance = distance_measure(exemplar, instance, **param_perm)
                    if distance <= min_distance or min_distance is None:
                        if distance < min_distance or min_distance is None:
                            closest_exemplar_indices.clear()
                            min_distance = distance
                        closest_exemplar_indices.append(exemplar_index)
                closest_exemplar_index = self.get_rand().choice(closest_exemplar_indices)
                splits[closest_exemplar_index][class_label].append(instance)


        gain = self._gain()
        return {
            self._exemplars_key: exemplars,
            self._param_perm_key: param_perm,
            self._gain_value_key: gain,
        }

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
        exemplars = {} # todo exemplar pick strategy
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
            if class_label not in exemplars.keys():
                exemplars[class_label] = []
            exemplars[class_label].append(exemplar)
        return exemplars, remaining_binned_instances

    def set_params(self, **params):
        super(ProximityTree, self).set_params(**params)
        super(ProximityTree, self)._set_param(self.gain_key, self.default_gain, params, prefix='_')
        super(ProximityTree, self)._set_param(self.r_key, self.default_r, params, prefix='_')
        super(ProximityTree, self)._set_param(self.param_pool_obtainer_key, self.default_param_pool_obtainer, params, prefix='_')
        super(ProximityTree, self)._set_param(self.stop_splitting_key, self.default_stop_splitting, params, prefix='_')

    def get_params(self):
        # todo get params
        return {}

if __name__ == "__main__":
    x_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    tree = ProximityTree(**{Randomised.rand_state_key: np.random.RandomState(3)})
    tree.fit(x_train, y_train)
