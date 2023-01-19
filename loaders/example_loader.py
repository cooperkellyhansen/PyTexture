import numpy as np
import pandas as pd
from kosh import KoshLoader


class ExampleLoader(KoshLoader):
    '''

    An example loader for Kosh based on a messy DREAM3D
    csv file

    '''

    types = {"csv" : ["numpy", ]}
    

    def open(self):
        '''

        Grab csv and build a dictionary

        '''
        df = pd.read_csv(fname, nrows=1, header=None)
        num_grains = df.iat[0,0]
        data = {}

    def extract(self):
        

    def list_features(self):
        feature_names = []
