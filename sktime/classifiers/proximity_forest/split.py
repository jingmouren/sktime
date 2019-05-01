from pandas import DataFrame, Series
from sklearn.preprocessing import normalize

from utils.classifier import Classifier
import numpy as np

from utils.utilities import Utilities
from utils.transformations import tabularise


class Split(Classifier):

    def __init__(self,
                 pick_exemplars_method = None,
                 param_perm = None,
                 gain_method = None,
                 rand = np.random.RandomState):
        super(Classifier, self).__init__()
        self.param_perm = param_perm
        self.gain_method = gain_method
        self.pick_exemplars_method = pick_exemplars_method
        self._rand = rand
        # vars set in the fit method
        self._distances = None
        self.exemplar_instances = None
        self.exemplar_class_labels = None
        self.remaining_instances = None
        self.remaining_class_labels = None
        self.branch_instances = None
        self.branch_class_labels = None
        self._unique_class_labels = None
        self.distance_measure_param_perm = None
        self.distance_measure = None
        self.gain = None

    @staticmethod
    def get_distance_measure_key():
        return 'dm'

    def fit(self, instances, class_labels):
        if not isinstance(self.param_perm, dict) or not callable(self.param_perm): # todo empty?
            raise ValueError("parameter permutation must be a dict or callable to obtain dict")
        if not callable(self.gain_method):
            raise ValueError("gain method must be callable")
        if not callable(self.pick_exemplars_method):
            raise ValueError("gain method must be callable")
        if not isinstance(self._rand, np.random.RandomState):
            raise ValueError('rand not set to a random state')
        if callable(self.param_perm):
            self.param_perm = self.param_perm(instances)
        if self.distance_measure is None:
            key = self.get_distance_measure_key()
            self.distance_measure = self.param_perm[key]
            del self.param_perm[key]
        self.exemplar_instances, self.exemplar_class_labels, self.remaining_instances, self.remaining_class_labels = \
            self.pick_exemplars_method(instances, class_labels, self._rand)
        distances = self.exemplar_distances(self.remaining_instances)
        self.exemplar_class_labels = []
        num_exemplars = len(self.exemplar_instances)
        self.branch_instances = [None] * num_exemplars
        num_instances = instances.shape[0]
        for instance_index in np.arange(num_instances):
            exemplar_distances = distances[instance_index]
            instance = instances.iloc[instance_index, :]
            class_label = class_labels[instance_index]
            closest_exemplar_index = Utilities.arg_min(exemplar_distances, self._rand)
            self.branch_instances[closest_exemplar_index].append(instance)
            self.exemplar_class_labels[closest_exemplar_index].append(class_label)
        self.gain = self.gain_method(class_labels, self.branch_class_labels)
        self._unique_class_labels = np.unique(class_labels)
        return self

    def exemplar_distances(self, instances):
        if not isinstance(instances, DataFrame):
            raise ValueError("instances not in panda dataframe")
        num_instances = instances.shape[0]
        num_exemplars = len(self.exemplar_instances)
        overall_distances = np.zeros((num_instances, num_exemplars))
        for instance_index in np.arange(num_instances):
            instance = instances.iloc[instance_index, :]
            distances = self.exemplar_distance_inst(instance)
            overall_distances[instance_index] = distances
        return overall_distances

    def exemplar_distance_inst(self, instance):
        if not isinstance(instance, Series):
            raise ValueError("instance not a panda series")
        num_exemplars = len(self.exemplar_instances)
        distances = np.zeros(num_exemplars)
        for exemplar_index in np.arange(num_exemplars):
            exemplar = self.exemplar_instances[exemplar_index]
            distance = self._find_distance(exemplar, instance)
            distances[exemplar_index] = distance
        return distances

    def predict_proba(self, instances):
        if not isinstance(instances, DataFrame):
            raise ValueError("instances not in panda dataframe")
        num_instances = instances.shape[0]
        num_exemplars = len(self.exemplar_instances)
        num_unique_class_labels = len(self._unique_class_labels)
        distributions = np.empty((num_instances, num_unique_class_labels))
        distances = self.exemplar_distances(instances)
        for instance_index in np.arange(num_instances):
            distribution = np.zeros(num_unique_class_labels)
            distributions[instance_index] = distribution
            for exemplar_index in np.arange(num_exemplars):
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