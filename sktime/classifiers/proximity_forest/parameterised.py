import warnings


class Parameterised:
    def __init__(self, **params):
        if type(self) is Parameterised:
            raise Exception('this is an abstract class')
        self.set_params(**params)
        super(Parameterised, self).__init__()

    def get_params(self):
        raise Exception('this is an abstract class')

    def set_params(self, **params):
        raise Exception('this is an abstract class')

    def _set_param(self, name, default_value, params, prefix='', suffix=''):
        varname = prefix + name + suffix
        try:
            value = params[name]
        except:
            s = "using default value for " + str(name) + ": "
            if callable(default_value):
                s += default_value.__name__
            else:
                s += str(default_value)
            warnings.warn(s, stacklevel=3)
            value = default_value
        setattr(self, varname, value)