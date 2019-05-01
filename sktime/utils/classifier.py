from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from utils import utilities
from utils.utilities import check_data


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 rand=np.random.RandomState()):
        super(BaseEstimator, self).__init__()
        self.rand = rand

    def predict_proba(self, instances):
        raise NotImplementedError('abstract method')

    def predict(self, instances):
        check_data(instances)
        distributions = self.predict_proba(instances)
        predictions = np.empty((distributions.shape[0]))
        for instance_index in range(0, predictions.shape[0]):
            distribution = distributions[instance_index]
            prediction = utilities.max(distribution, self.rand)
            predictions[instance_index] = prediction
        return predictions

