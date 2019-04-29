import warnings

import numpy as np
from classifiers.proximity_forest.parameterised import Parameterised


class Randomised(Parameterised):

    rand_state_key = 'rand_state'
    init_rand_state_key = 'init_rand_state'
    rand_key = 'rand'

    def __init__(self, **params):
        self.__rand = None
        self._init_state = None
        super(Randomised, self).__init__(**params)

    def set_params(self, **params):
        rand_state = params.get(self.rand_state_key)
        if isinstance(rand_state, int):
            rand_state = np.random.RandomState(rand_state).get_state()
        if rand_state is None:
            rand_state = params.get(self.init_rand_state_key)
        if rand_state is None:
            warnings.warn('no random state given, default to seed 0')
            rand_state = np.random.RandomState(0).get_state()
        self.__rand = np.random.RandomState() # todo straighten this out, bit confusing as is, errs out with no param
        self.__rand.set_state(rand_state)
        self._init_state = rand_state

    def get_rand(self):
        return self.__rand

    def get_params(self):
        return {self.init_rand_state_key: self._init_state, self.rand_state_key: self.__rand.get_state()}