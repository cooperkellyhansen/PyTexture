import HCPTexture,FCCTexture

import math
from numpy import *
import pandas as pd
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cosine as cos_sim
from HCPTexture import *
from FCCTexture import *

pd.options.display.max_colwidth = 1500 # Some strings are long so this will fix truncation

class SVE:

    '''
    This class is a representaion of an SVE (Statistical Volume Element). The object is built from
    csv files that are output by DREAM.3D software. There are methods to clean data, grab data, and manipulate data.
    it also uses the Orientation and FCCGrain classes and FCCTexture wrapper class written by Dr. Jacob Hochhalter
    '''


    #############################################
    def __init__(self):
        '''
        Many of these SVE attributes are dictionaries. The key of the dictionary corresponds to the grain number (Feature_ID).
        The attributes are based off the output options of the DREAM.3D software and can be added to for various applications.
        '''

        self.num_grains = int
        self.num_elems = 27000 # standard for DREAM.3D SVEs
        self.sve_num = int
        self.alpha = float
        self.beta = float
        self.sve_taylor_factor = float # Single Taylor factor of SVE
        self.grain_taylor_factor = {} # Taylor factor of each grain
        self.sve_principal = [] # principal stress and strain of SVE [S1,S2,S3,E1...]
        self.grain_principal = {} # principal stress and strain of each grain
        self.strain_FIPs = {} # max cyclic strain/2. These are FIPs calculated without the stress normal term on small strain FEM.
        self.micro_FIPs = {} # FIPs calculated from small strain FEM
        self.elemental_strain_FIPs = {} # strain FIPs per element
        self.elemental_micro_FIPs = {} # micro FIPs per element

        self.elem_grain_link = {}
        self.grain_elem_stress = {} # all stress tensors for elements in grain
        self.grain_elem_strain = {}
        self.grain_boundary_nodes = {}
        self.fip_dist_to_gb = {}
        self.fip_dist_to_gb_sphere = {}
        self.node_coords = {}
        self.elem_coords = {}
        self.vAvg_stress = {}
        self.vAvg_strain = {}
        self.strain_range = {}
        self.num_elem_neighbors = {}
        self.textureTensor = []

        self.shared_surface_area = {}
        self.neighbor_misorient = {}
        self.neighbor_mp = {}
        self.neighbor_shared_surface_area = {}

        self.quaternions = {}
        self.axis_euler = {}
        self.euler = {}
        self.centroid = {}
        self.volume = {}
        self.omega3s = {}
        self.axislengths = {}
        self.max_fips = {}  # max fip per grain
        self.max_band_fips = {} #max fip per averaged band
        self.taylorFactor = {}

        self.grain_neighbor_link = {}

        self.sveHCPTexture = HCPTexture()
        self.sveFCCTexture = FCCTexture()

        #all texture attributes are found in the FCC texture class
    def construct_SVE(self,fname1,fname2):
        # TODO: use the other functions to fill the rest of the attributes in this class
        return None

    #the first step is to build a bunch of orientation objects from the euler angles or quaternions
    #this can be done with the HCP or FCC texture classes
    def textureFromEulerAngles(self,fname,structure='FCC'):
        '''
        This is a function to create a texture object from euler angles using the Texture classes.

        :param fname: SVE feature data file. This is in the form of a csv.
        :param structure: The grain structure type of the SVE. This supports FCC and HCP through those subclasses.
        :return: None
        '''

        # Store number of grains in SVE
        df = pd.read_csv(fname,nrows=1,header=None)
        self.num_grains = df.iat[0, 0]
        self.sve_num = [char for char in fname if char.isdigit()]

        # Pull out euler angles
        df = pd.read_csv(fname, header=1, skiprows=0, nrows=self.num_grains)
        df = df.loc[:,['EulerAngles_0','EulerAngles_1','EulerAngles_2']]

        # Convert from csv to txt
        df.to_csv('EulerAngles.txt',header=None, index=None, sep=' ',mode='w+')

        # Build texture object from the file
        with open('EulerAngles.txt', 'r') as f:
            if structure == 'HCP':
                self.sveHCPTexture.fromEulerAnglesFile(f)
            elif structure == 'FCC':
                self.sveFCCTexture.fromEulerAnglesFile(f)

    #############################################
    def set_features(self,fname):
        '''
        This is a function to set the attributes of an SVE object that are found in the output feature data file
        each attribute is set as a dictionary with grain numbers as the keys.

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Store number of grains
        df = pd.read_csv(fname, nrows=1, header=None)
        self.num_grains = df.iat[0, 0]
        self.sve_num = [char for char in fname if char.isdigit()]

        # Read first third of csv
        feature_data = pd.read_csv(fname, header=1, skiprows=0, nrows=self.num_grains)
        feature_data.set_index('Feature_ID',inplace=True)

        # TODO: Cut out the 0 quaternions

        # Fill SVE attributes from first 1/3 of csv file
        for gnum in feature_data.index:
            # gnum = grain number
            # gname = grain name key
            gname = 'Grain_{}'

            self.quaternions[gname.format(gnum)] = feature_data.loc[gnum,['AvgQuats_0','AvgQuats_1','AvgQuats_2','AvgQuats_3']].tolist()
            self.axis_euler[gname.format(gnum)] = feature_data.loc[gnum,['AxisEulerAngles_0','AxisEulerAngles_1','AxisEulerAngles_2']].tolist()
            self.euler[gname.format(gnum)] = feature_data.loc[gnum,['EulerAngles_0','EulerAngles_1','EulerAngles_2']].tolist()
            self.centroid[gname.format(gnum)] = feature_data.loc[gnum,['Centroids_0','Centroids_1','Centroids_2']].tolist()
            self.volume[gname.format(gnum)] = feature_data.loc[gnum,'Volumes']
            self.omega3s[gname.format(gnum)] = feature_data.loc[gnum,'Omega3s'].tolist()
            self.axislengths[gname.format(gnum)] = feature_data.loc[gnum,['AxisLengths_0','AxisLengths_1','AxisLengths_2']].tolist()


    #############################################
    def set_grain_neighbors(self,fname):
        '''
        This function grabs the feature ID of the grains and their neighbors

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Store number of grains
        df = pd.read_csv(fname, nrows=1, header=None)
        self.num_grains = df.iat[0, 0]
        grains_and_neighbors = pd.read_csv(fname, skiprows=(self.num_grains+3), nrows=self.num_grains, header=None, sep='\n')

        # Fill attribute dictionary
        for grain in grains_and_neighbors.index:
            neighbors_cur = grains_and_neighbors.iloc[grain].to_string().split(',')
            del(neighbors_cur[0:2])
            self.grain_neighbor_link['Grain_{}'.format(grain + 1)] = list(map(int,neighbors_cur))

    #############################################
    def set_surface_area(self,fname):
        '''
        This function grabs the shared surface area between grains and their neighbors. It should be noted that
        the attribute dictionary is full of surface areas only and the labels of the neighboring grains should
        be pulled from the 'grain_neighbor_link' attribute.

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Store number of grains
        df = pd.read_csv(fname, nrows=1, header=None)
        self.num_grains = df.iat[0, 0]

        # Read shared surface area (ssa) portion of feature data
        ssa_data = pd.read_csv(fname, skiprows=((2 * self.num_grains) + 4), nrows=self.num_grains, header=None, sep='\n')

        # Fill attribute dictionary
        for grain in ssa_data.index:
            ssa_cur = ssa_data.iloc[grain].to_string().split(',')
            del(ssa_cur[0:2])
            self.shared_surface_area['Grain_{}'.format(grain + 1)] = list(map(float,ssa_cur))

        # label with grain neighbors
        for grain in self.shared_surface_area:
            neighbor_labels = {}
            for i,neighbor in enumerate(self.grain_neighbor_link[grain]):
                neighbor_labels[neighbor] = self.shared_surface_area[grain][i]
            self.neighbor_shared_surface_area[grain] = neighbor_labels

    #############################################
    def set_sub_band_data(self,fname):
        '''
        This is a function to retrieve the sub band max FIPs and store them in the
        appropriate attribute. Make sure num_grains is set before this

        :param fname: SVE feature data file. This is in the form of a csv.
        :return: None
        '''

        # Read data
        df = pd.read_csv(fname)
        df.set_index('grain',inplace=True)

        # Fill attribute dictionary
        for grain in df.index:
            self.max_fips['Grain_{}'.format(grain)] = df.loc[grain, 'FIP'].tolist()

    ############################################# 
    def set_band_data(self, fname):
        '''
        Set values of max band averaged FIPs for the SVE

        '''
        # Read data
        df = pd.read_csv(fname,header=0)
        max_bandAvg_fip = df.groupby(['grain']).max()

        #fill attribute dictionary
        self.max_band_fips = max_bandAvg_fip.to_dict()
    
    #############################################
    def set_grain_element_data(self,fname):
        '''
        Link elements and their FIP with the corresponding grain

        :param fname
        :return: None
        '''
        # Grab elements and FIPs
        df = pd.read_csv(fname,usecols=['element','grain_id','FIPs'])
        # link and fill dictionary
        df = df.groupby('grain_id').agg({'element': list, 'FIPs': list})
        self.elem_grain_link = df.to_dict()


    #############################################
    def calc_misorientations(self,structure='FCC'):
        '''
        Calculate the maximum misorientations between all pairs of grains
        (neighboring or not) using the texture and grain classes. Then sort
        according to neighbors.
        :param structure: Grain structure of the SVE
        :return: None
        '''

        # Calculate the maximum misorientations from the texture class
        if structure == 'HCP':
            self.sveHCPTexture.calc_misorient()

            # Pair the actual neighbors and save
            for grain in self.sveHCPTexture.misorient:
                neighbor_labels = {}
                for neighbor in self.grain_neighbor_link[grain]:
                    neighbor_labels[neighbor] = self.sveHCPTexture.misorient[grain][neighbor - 1]
                self.neighbor_misorient[grain] = neighbor_labels

        elif structure == 'FCC':
            self.sveFCCTexture.calc_misorient()

            # Pair the actual neighbors and save
            for grain in self.sveFCCTexture.misorient:
                neighbor_labels = {}
                for neighbor in self.grain_neighbor_link[grain]:
                    neighbor_labels[neighbor] = self.sveFCCTexture.misorient[grain][neighbor - 1]
                self.neighbor_misorient[grain] = neighbor_labels

        # create pickle file for easy access.
        with open('neighbor_misorient.pkl', 'wb') as f:
            pickle.dump(self.neighbor_misorient,f)

    #############################################
    def calc_schmidFactors(self, structure='FCC',file_type='rod'):
        '''
        This function calculates the max Schmid factors (global) using the grain class
        the slip transmission (m') is also calculated here. A primary slip dictionary
        that stores the schmid factor, the normal to the plane, and the slip direction
        is created as well as a dictionary of m' values. A pickle file of m' values is
        also created for ease of use.

        :param fname: A desired file name. The file object is created here.
        :param structure: Grain structure of SVE
        :return: None
        '''

        # TODO: the rodrigues vectors are not being calculated correctly. It is causing a error in the Schmid factor calc
        # Open a file for max Schmid factors
        with open('schmidFactors.txt', 'w') as f:
            # Use texture class to calculate the Schmid factors
            # according to grain structure.

            if structure == 'HCP':
                if file_type == 'rod':
                    self.sveHCPTexture.toRodriguesFile2(f)
                elif file_type == 'euler':
                    self.sveHCPTexture.toEulerAnglesFile2(f)
            elif structure == 'FCC':
                if file_type == 'rod':
                    self.sveFCCTexture.toRodriguesFile2(f)
                elif file_type == 'euler':
                    self.sveFCCTexture.toEulerAnglesFile2(f)

        # grab all schmid factors for further analysis
    #############################################
    def calc_mPrime(self,structure_type='FCC',to_file=False):
        # determine texture
        if structure_type == 'HCP':
            texture = self.sveHCPTexture
        else:
            texture = self.sveFCCTexture

        # Calculate the maximum misorientations from the texture class
        texture.calc_mPrime()

        # Pair the actual neighbors and save
        for grain in texture.mp:
            neighbor_labels = {}
            try:
                for neighbor in self.grain_neighbor_link[grain]:
                    neighbor_labels[neighbor] = texture.mp[grain][neighbor - 1]
                    print(len(texture.mp[grain]))
                self.neighbor_mp[grain] = neighbor_labels
            except IndexError:
                print('Index Error @: ', neighbor - 1)

        # save to pickle file
        if to_file:
            with open('mprime.pkl', 'wb') as f:
                pickle.dump(self.neighbor_mp, f)

    #############################################
    def calc_fatigueLife(self,FIP):

        # TODO: find experimental values for IN625
        N = 0.0 # fatigue life of current band
        D_st = 0.0  # diameter of current band being evaluated
        phi = 0.0 # mechanical irreversibility at crack tip
        A = 0.0 # experimental constant
        b = 0.0 # experimental constant
        CTD = A*(FIP)**b # crack tip displacement

        #influence of neighboring grains
        n = None # number of neighboring bands. Calculate in CP-FE
        D_nd = None # diameter of neighboring bands
        theta_dis = None # angle of disorientation between two neighboring bands
        omega = 1 - theta_dis / 20 # disorientation factor
        influence_ng = [omega[neighbor] * D_nd[neighbor] for neighbor in n]

        # Beta term
        d_ref_gr = 0.0 # mean grain size of material (IN625)
        beta = (D_st + influence_ng) / d_ref_gr

        # constants
        c1 = phi * (beta * A * (FIP)**b - CTD)
        c2 = (phi * 2 * beta * A * (FIP)**b) / ((D_st + influence_ng)**2)

        # fatigue life of current band
        N.append((1 / np.sqrt(c1 * c2)) * np.arctanh(D_st * np.sqrt(c1/c2)))


        return None


    #############################################
    def calc_volumeAvg(self,fname_ss,fname_link):

        # Make dataframe for ease of use
        data = pd.read_csv(fname_ss,header=0)

        # Read shared surface area (ssa) portion of feature data
        elem_data = pd.read_csv(fname_link, header=None, sep='\n')

        # Fill attribute dictionary
        for grain in elem_data.index:
            elems_cur = elem_data.iloc[grain].to_string().split(',')
            elems_cur = [elem.strip() for elem in elems_cur]
            del(elems_cur[0])
            del(elems_cur[-1])
            self.grain_elem_link['Grain_{}'.format(grain + 1)] = list(map(int,elems_cur))

        # First build all of the matrices for each element

        # Max Peak

        strain_range=[]
        for grain in self.grain_elem_link:
            s_max = []
            e_max = []
            for elem in self.grain_elem_link[grain]:
                e_cur = np.empty((3, 3))
                s_cur = np.empty((3, 3))
                for j in range(0, 3):
                    if j == 1:
                        s_cur[j] = data.loc[elem+108000, ['S12', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+108000, ['Ep21', 'Ep22', 'Ep23']]  # current stress matrix
                    elif j == 2:
                        s_cur[j] = data.loc[elem+108000, ['S13', 'S23', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+108000, ['Ep31', 'Ep32', 'Ep33']]  # current strain matrix
                    else:
                        s_cur[j] = data.loc[elem+108000, ['S' + str(j + 1) + '1', 'S' + str(j + 1) + '2','S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+108000, ['Ep11', 'Ep12', 'Ep13']]  # current stress matrix
                e_max.append(e_cur)
                s_max.append(s_cur)

            s_min = []
            e_min = []
            for elem in self.grain_elem_link[grain]:
                e_cur = np.empty((3, 3))
                s_cur = np.empty((3, 3))
                for j in range(0, 3):
                    if j == 1:
                        s_cur[j] = data.loc[elem+81000, ['S12', 'S' + str(j + 1) + '2', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+81000, ['Ep21', 'Ep22', 'Ep23']]  # current stress matrix
                    elif j == 2:
                        s_cur[j] = data.loc[elem+81000, ['S13', 'S23', 'S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+81000, ['Ep31', 'Ep32', 'Ep33']]  # current strain matrix
                    else:
                        s_cur[j] = data.loc[elem+81000, ['S' + str(j + 1) + '1', 'S' + str(j + 1) + '2','S' + str(j + 1) + '3']]  # current stress matrix
                        e_cur[j] = data.loc[elem+81000, ['Ep11', 'Ep12', 'Ep13']]  # current stress matrix
                #e_min.append(max(np.linalg.eig(e_cur)[0], key=abs))  # max eigenvalue of strain in minimum
                e_min.append(e_cur)
                s_min.append(s_cur)  # max eigenvalue of stress in maximum

            sr = []
            for elem,strain in enumerate(e_max):
                sr.append(np.array(e_max[elem]) - np.array(e_min[elem]))
            self.strain_range[grain] = sr

            #self.grain_elem_stress[grain] = s
            #self.grain_elem_strain[grain] = e

            #self.vAvg_stress[grain] = np.mean(np.array(self.grain_elem_stress[grain]),axis=0)
            #self.vAvg_strain[grain] = strain_range

        return None

    #####################################
    def calc_fip_dist_from_boundary(self):

        # generate all element node coords
        node = 1
        # assume nodes start at [0,0,0] and increment by 2s.
        for z in range (0,51,2):
            for y in range (0,51,2):
                for x in range (0,51,2):
                    self.node_coords[node] = [x,y,z]
                    node += 1

        elem = 1
        # assuming element centroids start at [1,1,1] and go by 2s
        for z in range(1,50,2):
            for y in range(1,50,2):
                for x in range(1,50,2):
                    self.elem_coords[elem] = [x,y,z]
                    elem +=1            

        #build each element with node names
        elem_nodes = {} # {'1': [[node_name1], [node_name2]....]}
        val = list(self.node_coords.values())
        for elem,coords in self.elem_coords.items():
            cur = [] 
            #bottom nodes
            cur.append(val.index([coords[0]-1,coords[1]-1,coords[2]-1])+1)
            cur.append(val.index([coords[0]-1,coords[1]-1,coords[2]+1])+1)
            cur.append(val.index([coords[0]-1,coords[1]+1,coords[2]-1])+1)
            cur.append(val.index([coords[0]-1,coords[1]+1,coords[2]+1])+1)
            #top nodes
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            cur.append(val.index([coords[0]+1,coords[1]+1,coords[2]+1])+1)
            elem_nodes[elem] = cur

        # loop through grains in SVE
        for grain in range(1,self.num_grains+1):
            grain_node_set = set()
            # find all node names in grain
            grain_node_list = [elem_nodes[elem] for elem in self.elem_grain_link['element'][grain]]
            grain_node_list = [item for sublist in grain_node_list for item in sublist] #flatten
            # find the coords of elements in grain
            grain_elem_coord_list = [self.elem_coords[elem] for elem in self.elem_grain_link['element'][grain]]
            # update set
            grain_node_set.update(grain_node_list)
            # find all neighbors
            neighbors = [neighbor for neighbor in self.grain_neighbor_link['Grain_{}'.format(grain)]]
            # set for neighbor node names
            neighbor_node_set = set()
            neighbor_node_list_list = []
            for neighbor in neighbors:
                neighbor_node_list = [elem_nodes[elem] for elem in self.elem_grain_link['element'][neighbor]] # list of node names of neighbor
                neighbor_node_list = [item for sublist in neighbor_node_list for item in sublist] #flatten
                neighbor_node_set.update(neighbor_node_list)
                neighbor_node_list_list.append(neighbor_node_list)
                
            #boundary nodes
            self.grain_boundary_nodes['Grain_{}'.format(grain)] = grain_node_set.intersection(neighbor_node_set)

            # find element containing max FIP
            max_fip = max(self.elem_grain_link['FIPs'][grain])
            max_fip_elem = self.elem_grain_link['element'][grain][(self.elem_grain_link['FIPs'][grain]).index(max_fip)]
            # find coords of max FIP element
            max_fip_elem_coords = self.elem_coords[max_fip_elem]
            # find minimum distance to boundary nodes
            for elem in self.elem_grain_link['element'][grain]:
                min_dist = 10000
                for boundary_node in self.grain_boundary_nodes['Grain_{}'.format(grain)]:
                    dist = np.linalg.norm([[np.linalg.norm(i-j) for j in self.node_coords[boundary_node]] for i in self.elem_coords[elem]])
                    if abs(dist) < min_dist:
                        min_dist = dist
                self.fip_dist_to_gb['elem_{}'.format(elem)] = min_dist
                cent_dist = np.linalg.norm([[np.linalg.norm(i-j) for j in self.centroid['Grain_{}'.format(grain)]] for i in self.elem_coords[elem]])
                self.fip_dist_to_gb_sphere['elem_{}'.format(elem)] = abs(self.volume['Grain_{}'.format(grain)]/2 - abs(cent_dist))
    
    def texture_tensor(self):
        #standard coord of c-axis
        x,y,z = 0,0,1
        
        #loop through orientations 
        #rot  = R.from_euler('zxz', np.fromiter(self.sveFCCTexture.eulerDict.values(),dtype=float), degrees=False)
        #c_axis = rot.apply([x,y,z], inverse=True) # map grain orientation to c-axis
        #vol = np.fromiter(self.volume.values(),dtype=float)  
        #vol_tot = sum(self.volume.values())

        #calculate I
        #text_tens = np.tensordot(c_axis,c_axis*vol,axes=(0,0))/vol_tot
        
        #calculate scalar parameters
        numerator = []
        denom_r = []
        denom_l = []
        similarities = []
        for grain_num,orientation in self.sveFCCTexture.eulerDict.items():
            # (alpha) cosine similarity weighted by grain volume
            numerator.append(cos_sim(list(orientation), [x,y,z], w=[self.volume['Grain_{}'.format(grain_num+1)]]*3))
            denom_r.append(np.asarray(orientation)**2 * self.volume['Grain_{}'.format(grain_num+1)])
            denom_l.append(np.asarray([x,y,z])**2 * self.volume['Grain_{}'.format(grain_num+1)])
            
            # (beta) cosine similarity weighted by grain volume
            similarities.append(cos_sim(orientation, [x,y,z]))
        self.alpha = sum(numerator)/(np.sqrt(sum(denom_l))*np.sqrt(sum(denom_r)))

        avg_sim_weighted = []
        for grain_num,similarity in enumerate(similarities):
            avg_sim_weighted.append(similarity * self.volume['Grain_{}'.format(grain_num+1)])
        avg_sim_weighted = sum(avg_sim_weighted)/sum(list(self.volume.values()))

        num = []
        for grain_num,similarity in enumerate(similarities):
            num.append((similarity - avg_sim_weighted)**2)
        denom = ((len(self.volume.values())-1)/len(self.volume.values()))*sum(list(self.volume.values()))
        self.beta = np.sqrt(sum(num)/denom)
        
        #calculate the sigma_pred
    
    def FIP_neighbors(self,sve_num,loading_scenario):
        boundary_cells_vtk = ('IN625/Loading_Scenario_{}/reprocess_AFR62_{}{}/Output_FakeMatl_0.vtk'.format(loading_scenario,loading_scenario,sve_num))
        file = open(boundary_cells_vtk)
        content = file.readlines()
        vtk_boundary_cells=content[3932:4714]

        BC_list=[]
        el_list=[]
        for i in range (0, len(vtk_boundary_cells)):
            el=i+1
            line_i=vtk_boundary_cells[i].split()
            for j in range(len(line_i)):
                el = el+j
                el_list+=[el]
            BC_list+=line_i
        
        df = pd.DataFrame(list(zip(el_list, BC_list)), columns=['element', 'boundary_cells'])
        for grain in range(1, self.num_grains + 1):
            for elem in self.elem_grain_link['element'][grain]:
                self.num_elem_neighbors['elem_{}'.format(elem)] = df.at[elem-1,'boundary_cells']

    def taylor_factors(self, f, sve_num,loading_scenario):

        # open file and extract taylor factor for SVE
        df_sve = pd.read_csv('IN625/Loading_Scenario_{}/Case_{}/Case_{}_taylor_factors.csv'.format(loading_scenario,loading_scenario,loading_scenario))
        self.sve_taylor_factor = df_sve.at[sve_num-1, 't_factor']

        # open grain file and extract taylor factor for grains
        df_grains = pd.read_csv(f)
        for grain_num,tf in enumerate(df_grains['taylor_factor'], start=1):
            self.grain_taylor_factor['Grain_{}'.format(grain_num)] = tf
  
    def principal_stress_strain(self,f,sve_num,loading_scenario):

        # open file and extract principal information for SVE
        df_sve = pd.read_csv('IN625/Loading_Scenario_{}/Case_{}/{}_prin_stress_strain.csv'.format(loading_scenario,loading_scenario,loading_scenario),dtype=float)
        df_sve.drop(columns=df_sve.columns[0], axis=1, inplace=True)
        self.sve_principal = list(df_sve.iloc[sve_num-1, :])

        # open file and extract principal information for each grain
        df_grains = pd.read_csv(f,dtype=float)
        df_grains.drop(columns=df_grains.columns[0], axis=1, inplace=True)
        for i in range(len(df_grains)):
            self.grain_principal['Grain_{}'.format(i+1)] = list(df_grains.iloc[i,:])

    def small_strain_data(self, sve_num):
        
        # open file and extract small strain FEM simulation data (slip range data)
        df_slip = pd.read_csv('IN625/Loading_Scenario_A/small_strain_data/{}/slip_range_max_band_grain_FIPs.csv'.format(sve_num))
        df_slip.drop(columns=df_slip.columns[0], axis=1, inplace=True) # drop index column
        for i in range(len(df_slip)):
            self.strain_FIPs['Grain_{}'.format(i+1)] = df_slip.iloc[i,1]

        # open file and extract small strain FEM simulation data (micro FIP data)
        df_micro = pd.read_csv('IN625/Loading_Scenario_A/small_strain_data/{}/micro_FIP_max_band_grain_FIPs.csv'.format(sve_num))
        df_micro.drop(columns=df_micro.columns[0], axis=1, inplace=True) # drop index column
        for i in range(len(df_micro)):
            self.micro_FIPs['Grain_{}'.format(i+1)] = df_micro.iloc[i,1]
    
    def elemental_small_strain(self,sve_num):

        # open file and extract small strain elemental FIPs (slip range data)
        df_slip = pd.read_csv('IN625/Loading_Scenario_A/small_strain_data/{}/slip_range_final_updated_df.csv'.format(sve_num))
        for i in range(len(df_slip)):
            self.elemental_strain_FIPs['elem_{}'.format(df_slip['element'][i])] = (df_slip['max_SBA_FIPs'][i],df_slip['grain_id'][i])

        # open file and extract small strain elemental FIPs (micro FIP)
        df_micro = pd.read_csv('IN625/Loading_Scenario_A/small_strain_data/{}/slip_range_final_updated_df.csv'.format(sve_num))
        for i in range(len(df_micro)):
            self.elemental_micro_FIPs['elem_{}'.format(df_micro['element'][i])] = (df_micro['max_SBA_FIPs'][i],df_slip['grain_id'][i])
