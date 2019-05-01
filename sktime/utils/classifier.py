from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from utils.utilities import Utilities


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super(BaseEstimator, self).__init__()

    def predict_proba(self, instances):
        raise NotImplementedError('abstract method')

    def predict(self, instances):
        if not isinstance(instances, DataFrame):
            raise ValueError("instances not in panda dataframe")
        distributions = self.predict_proba(instances)
        predictions = np.empty((distributions.shape[0]))
        for instance_index in range(0, predictions.shape[0]):
            distribution = distributions[instance_index]
            prediction = Utilities.max(distribution, self._rand)
            predictions[instance_index] = prediction
        return predictions
