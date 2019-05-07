from joblib import Parallel
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils import utilities
from utils.utilities import check_data


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 rand=None,
                 ):
        super().__init__()
        self.rand = rand
        self.label_encoder = None

    def predict_proba(self, instances):
        raise NotImplementedError('abstract method')

    def predict(self, instances, should_check_data=True):
        if should_check_data:
            check_data(instances)
        distributions = self.predict_proba(instances, should_check_data=False)
        predictions = np.empty((distributions.shape[0]))
        for instance_index in range(0, predictions.shape[0]):
            distribution = distributions[instance_index]
            prediction = utilities.max(distribution, self.rand)
            predictions[instance_index] = prediction
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

