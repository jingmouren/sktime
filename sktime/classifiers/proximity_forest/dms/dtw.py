import warnings

from pandas import DataFrame

from classifiers.proximity_forest.dms.distance_measure import DistanceMeasure
import numpy as np

from distances import dtw_distance


class Dtw(DistanceMeasure):

    delta_key = 'delta'
    default_delta = 0

    def __init__(self, **params):
        self._delta = None
        super(Dtw, self).__init__(**params)

    def distance(self, a, b, cut_off):
        return dtw_distance(a, b, 0) # **self.get_params())

    def set_params(self, **params):
        super(Dtw, self)._set_param(self.delta_key, self.default_delta, params, '_')

    def get_params(self):
        return {self.delta_key: self._delta}

if __name__ == "__main__":
    dtw = Dtw(delta=5)
    print('hello')
    a = np.array([1,2,3], dtype=float)
    b = np.array([1,5,3], dtype=float)
    a = a.reshape((a.shape[0], 1))
    b = b.reshape((b.shape[0], 1))
    dist = dtw.distance(a, b, 100)
    print(dist)
