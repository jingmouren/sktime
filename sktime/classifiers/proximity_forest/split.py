from scipy.stats import rv_continuous, rv_discrete
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize

from classifiers.proximity_forest.randomised import Randomised
import numpy as np

from classifiers.proximity_forest.utilities import Utilities
from utils.transformations import tabularise


class Split(BaseEstimator):

    def __init__(self,
                 pick_exemplars_method = None,
                 param_perm = None,
                 gain_method = None,
                 rand = np.random.RandomState):
        self.param_perm = param_perm
        self.gain_method = gain_method
        self.pick_exemplars_method = pick_exemplars_method
        self._rand = rand
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

    # todo mixin
    # todo checks
    # todo fit return self
    # todo python built in methods for class
    # todo score
    # todo diff between static method + class method with cls param - is that just self?
    # todo return class label from predict, not index!

    @staticmethod
    def get_distance_measure_key():
        return 'dm'

    def fit(self, instances, class_labels):
        # todo none checks
        if callable(self.param_perm):
            self.param_perm = self.param_perm(instances)
        if self.distance_measure is None:
            key = self.get_distance_measure_key()
            self.distance_measure = self.param_perm[key]
            del self.param_perm[key]
        self.exemplar_instances, self.exemplar_class_labels, self.remaining_instances, self.remaining_instances = \
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
        num_instances = instances.shape[0]
        num_exemplars = len(self.exemplar_instances)
        overall_distances = np.zeros((num_instances, num_exemplars))
        for instance_index in np.arange(num_instances):
            instance = instances.iloc[instance_index, :]
            distances = np.zeros(num_exemplars)
            overall_distances[instance_index] = distances
            for exemplar_index in np.arange(num_exemplars):
                exemplar = self.exemplar_instances[exemplar_index]
                distance = self._find_distance(exemplar, instance)
                distances[exemplar_index] = distance
        return overall_distances

    def predict_proba(self, instances):
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

    def predict(self, instances):
        num_instances = instances.shape[0]
        distributions = self.predict_proba(instances)
        predictions = np.empty(num_instances)
        for instance_index in range(0, num_instances):
            distribution = distributions[instance_index]
            prediction = Utilities.arg_max(distribution, self._rand)
            predictions[instance_index] = prediction
        return predictions

    def _find_distance(self, instance_a, instance_b):
        instance_a = tabularise(instance_a, return_array=True)
        instance_b = tabularise(instance_b, return_array=True)
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        return self.distance_measure(instance_a, instance_b, **self.distance_measure_param_perm)