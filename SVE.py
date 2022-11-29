import FCCTexture

from numpy import *
import pandas as pd
import pickle
from FCCTexture import *

pd.options.display.max_colwidth = 1500 # Some strings are long so this will fix truncation

class DeformedFCCPolycrystal(FCCTexture):

    '''
    This class is a representation of an SVE (Statistical Volume Element) that has undergone CPFEM. The texture object is first built
    in the FCCTexture class from the DREAM3D data. It can be considered a wrapper class for the FCCTexture class where this polycrystal
    has been deformed. 
    This was originally written to explore microstructure attributes and given data from CP-FEM simulations (stress and strain tensors) 
    one can calculate fatigue indicator parameters and eventually fatigue life based on microstructure.
    '''

    #############################################
    def __init__(self, num_grains, grain_orientations_file, orientation_type):
        '''
        Many of these SVE attributes are dictionaries. The key of the dictionary corresponds to the grain number
        which here always starts at 1 (i.e. 'Grain_{#}').
        The attributes are based off the output options of the DREAM.3D software and can be added to for various applications.
        '''
        
        self.NUM_GRAINS = num_grains # number of grains in the SVE
        self.ORIENTATION_TYPE = orientation_type # metric used to orient grains of SVE (i.e. quaternions, eulers, etc.).

        self.grain_orientations_file = grain_orientations_file # file containing the orientations of each grain
        self.quaternions = {} # quaternion representation of grain orientations 
        self.axis_euler_angles = {} # axis euler angle representation of grain orientations 
        self.euler_angles = {} # euler angle representation of grain orientations 
        self.rodrigues_vectors = {} # rodrigues vector representation of grain orientations 


        self.centroid = {}
        self.volume = {}
        self.omega3s = {}
        self.axislengths = {}
        self.shared_surface_area = {}

        self.grain_neighbor_link = {} # grains and their respective grain boundary sharing neighbors
        self.elem_grain_link = {} # elements and the grains that they are a part of 

        
