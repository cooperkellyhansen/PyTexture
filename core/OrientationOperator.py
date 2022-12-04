import kosh
import numpy as np


class koshOrientation(kosh.KoshOperator):
    types = {'numpy': ['numpy', ]}

    def __init__(self, *args, **kwargs):

        super(koshOrientation, self).__init__(*args, **kwargs)
        self.options = kwargs
    def operate(self, *inputs, **kargs):

