import FCCGrain
from joblib.numpy_pickle_utils import xrange

from numpy import *
from FCCGrain import *

ZERO_TOL = 1e-8
R2D = 180.0 / pi
RD = array([1., 0., 0.])
ND = array([0., 0., 1.])
TD = array([0., 1., 0.])


class FCCTexture:
    '''
    FCCTexture is a wrapper class for the FCCGrain class.
    FCCTexture defines members and methods to organize/generate/modify
    more than two FCCGrain objects at a time, e.g. a polycrystal.
    '''

    ##############################################
    def __init__(self, ortho=True):
        '''
        The attributes here are the microstructure attributes you would find
        in the grain structure of a polycrystal.

        '''

        self.orientDict = {} # a dictionary of Orientation objects 
        self.eulerDict = {}
        self.rodDict = {}
        
        self.neighbors = {}
        self.otherNeighbors = {}
        self.misorient = {}
        self.primary_slip = {}
        self.mp = {}
        self.attrData = {}

        self.has_ortho_symm = ortho

        self.fromEulers = False
        self.fromRodrigues = False

    ##############################################
    def addGrain(self, fcco_grain, grain_name):
        '''
        add an FCCGrain object to the FCCTexture object
        
        fcco_grain = FCCGrain object
        grain_name = any hashable type to be used as dictionary key

        '''

        self.orientDict[grain_name] = fcco_grain

    ##############################################
    def fromEulerAnglesFile(self, fname, cols=(0, 1, 2), conv='passive', units='rad'):
        '''
        reads a file containing Bunge-Euler angles
        fname = file name string
        cols = tuple of columns in file that contain phi1, PHI, phi2 (numbering starts at 0)
        conv = 'active' or 'passive' (for FCCGrain object)
        units = 'rad' for radian input, 'deg' for degree input

        Note: the input file can have any number of columns,
              as long as 3 columns contain the euler angles
        '''

        self.fromEulers = True
        data = loadtxt(fname, comments='#', usecols=cols)
        if units == 'deg':
            data *= pi / 180.0
        key = 0
        for eulers in data:
            self.orientDict[key] = FCCGrain(conv=conv, ortho=self.has_ortho_symm)
            self.orientDict[key].fromEulerAngles(eulers)
            self.eulerDict[key] = eulers
            key += 1
            ##############################################

    def fromRodriguesFile(self, fname, conv='passive', num_atts=0):
        '''
        reads a file containing Rodrigues parameters
        fname = file name string
        conv = 'active' or 'passive' (for FCCGrain object)

        Note: the input file can have any number of columns,
              as long as the first 3 columns are the rodrigues
              parameters
        '''

        self.fromRodrigues = True
        if num_atts >= 0:
            data = loadtxt(fname, comments='#', usecols=(0, 1, 2))
            if num_atts > 0:
                cols = tuple(range(3, 3 + num_atts, 1))
                attr_data = loadtxt(fname, comments='#', usecols=cols)
        else:
            print("Can't have a negative number of attributes!")

        key = 0
        for rod in data:
            self.orientDict[key] = FCCGrain(conv=conv, ortho=self.has_ortho_symm)
            self.orientDict[key].fromRodrigues(rod)
            self.rodDict[key] = rod
            self.attrData[key] = attr_data[key]
            key += 1
    ##############################################
    def toEulerAnglesFile(self, fd, out='rad'):
        '''
        writes a file containing Bunge-Euler angles
        fd = open file object
        out = 'rad' or 'deg'
        '''

        for theta in self.orientDict:
            if self.fromEulers:
                # if the input data came as Euler Angles,
                # then write the same ones, not equivalents.
                phi1, PHI, phi2 = self.eulerDict[theta]
            else:
                phi1, PHI, phi2 = self.orientDict[theta].asEulerAngles()

            if (phi1, PHI, phi2) != (None, None, None):
                if out == 'rad':
                    line = "%0.4f %0.4f %0.4f\n" % (phi1, PHI, phi2)
                    fd.write(line)
                elif out == 'deg':
                    line = "%0.4f %0.4f %0.4f\n" % (phi1 * R2D, PHI * R2D, phi2 * R2D)
                    fd.write(line)
                else:
                    print("error: out must be 'rad' or 'deg'")
            else:
                print("error getting euler angles for angle %0.2f" % theta)

    ##############################################
    def toEulerAnglesFile2(self, fd, out='rad', load_vec=ND):
        '''
        writes a file containing Bunge-Euler angles with
        3 additional columns: max Schmid factor, nearness to
        the ND-rotated cube orientation, and nearness to planar
        double slip orientation
        fd = open file object
        out = 'rad' or 'deg'
        '''

        for idx,theta in enumerate(self.orientDict):

            #min_nd_rotated = 1. / ZERO_TOL
            #min_planar_double = 1. / ZERO_TOL

            if self.fromEulers:
                phi1, PHI, phi2 = self.eulerDict[theta]
            else:
                phi1, PHI, phi2 = self.orientDict[theta].asEulerAngles()

            cur_orient = self.orientDict[theta]
            m,n,d = cur_orient.maxSchmidFactor(load_vec)
            # find min angle slip direction
            self.primary_slip['Grain_{}'.format(idx + 1)] = [m,n,d]

            #min_nd_rotated = cur_orient.getAlignment(FCCGrain.families[100], ND)
            #min_planar_double = cur_orient.getAlignment(FCCGrain.families[111], TD)

            if (phi1, PHI, phi2) != (None, None, None):
                if out == 'rad':
                    line = "%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n" % (
                    phi1, PHI, phi2, m, n[0],n[1],n[2],d[0],d[1],d[2]) #min_nd_rotated, min_planar_double)
                    fd.write(line)
                else:
                    line = "%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n" % (
                    phi1 * R2D, PHI * R2D, phi2 * R2D, m, n[0],n[1],n[2],d[0],d[1],d[2]) #min_nd_rotated, min_planar_double)
                    fd.write(line)
            else:
                print("error getting euler angles for angle %0.2f" % theta)

    ##############################################
    def toRodriguesFile(self, fd, out='rad'):
        '''
        writes the Rodrigues parameters for each orientation
        Note: it is usually helpful to call "collapseToRodriguesFundamental" prior
        to writing the data.
        '''

        for theta in self.orientDict:
            if self.fromRodrigues:
                r1, r2, r3 = self.rodDict[theta]
            else:
                r1, r2, r3 = self.orientDict[theta].asRodrigues()

            if (r1, r2, r3) != (None, None, None):
                line = "%0.4f %0.4f %0.4f\n" % (r1, r2, r3)
                fd.write(line)
            else:
                print("error getting euler angles for angle %0.2f" % theta)

    ##############################################
    def toRodriguesFile2(self, fd):
        '''
        writes the Rodrigues parameters for each orientation with
        3 additional columns: max Schmid factor, nearness to
        the ND-rotated cube orientation, and nearness to planar
        double slip orientation

        Note: it is usually helpful to call "collapseToRodriguesFundamental" prior
        to writing the data.
        '''

        #        rod_node_stress = [
        #    1005, 1214, 1082, 1202, 1183, 1057, 1185, 1066, 1076, 1061,
        #    1213, 1004, 1081, 1191, 1289, 1245, 1101, 1229, 1290, 1005,
        #    1079, 1201, 1213, 1182, 1054, 1187, 1211, 1004, 1081, 1192,
        #    1293, 1243, 1101, 1241, 1287, 1245, 1265, 1301, 1186, 1182,
        #    1212, 1222, 1124, 1029, 1265, 1298, 1018, 1289, 1245, 1296,
        #    1179, 1268, 1183, 1206, 1222, 1264, 1243, 1302, 1231, 1287,
        #    1005, 1020, 1201, 1290, 1004, 1213, 1081, 1194, 1183, 1057,
        #    1186, 1067, 1061, 1214, 1082, 1101, 1293, 1004, 1081, 1193,
        #    1211, 1182, 1054, 1185, 1213, 1005, 1079, 1201, 1290, 1245,
        #    1101, 1242, 1289, 1243, 1265, 1298, 1184, 1182, 1212, 1215,
        #    1126, 1029, 1265, 1301, 1020, 1287, 1243, 1302, 1182, 1264,
        #    1183, 1206, 1225, 1268, 1296, 1018, 1293, 1239,  994, 1049,
        #    991, 1214, 1240, 1160, 1265, 1238,  994, 1213, 1240, 1160,
        #    1265, 1240,  994, 1048,  991, 1211, 1238, 1160, 1264, 1240,
        #    994, 1213, 1239, 1160, 1268
        #     ]

        #min_distance = 1. / ZERO_TOL
        #norm_min_r = 1. / ZERO_TOL

        for idx,theta in enumerate(self.orientDict):
            cur_orient = self.orientDict[theta]
            if self.fromRodrigues:
                r1, r2, r3 = self.rodDict[theta]
            else:
                r1, r2, r3 = cur_orient.asRodrigues()

            m,n,d = cur_orient.maxSchmidFactor(RD)
            # find min angle slip direction
            self.primary_slip['Grain_{}'.format(idx + 1)] = [m,n,d]

            #min_nd_rotated = cur_orient.getAlignment(FCCGrain.families[100], ND)
            #min_planar_double = cur_orient.getAlignment(FCCGrain.families[111], TD)

            if (r1, r2, r3) != (None, None, None):
                if (r1 > -ZERO_TOL) and (r2 > -ZERO_TOL):
                    line = "%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n" % (r1, r2, r3, m,n[0],n[1],n[2],d[0],d[1],d[2]) #min_nd_rotated, min_planar_double)
                    fd.write(line)
            else:
                print("error getting rodrigues parameters for angle %0.4f" % theta)

    ##############################################
    def toPointPFFiles(self, fd, family, spec_axis, num_atts=0):
        '''
        writes the pole figure coordinates for each FCCGrain object
        for the specified family and specimen axis
        family = FCCGrain.families[family]
        spec_axis = RD, TD, or ND
        '''

        min_distance = 1. / ZERO_TOL

        for theta in self.orientDict:

            cur_orient = self.orientDict[theta]

            data_pts = cur_orient.getPoleFigureCoords(FCCGrain.families[family], spec_axis)

            for pt in data_pts:
                line = "%0.4f %0.4f %0.4f" % (pt[0], pt[1], 0.0)
                if num_atts > 0:
                    for c in self.attrData[theta]: line += " %0.4f" % c
                fd.write(line + "\n")

        return 1

    ##############################################
    def toContourPFFiles(self, fd, family, spec_axis, binsize=pi / 36.):
        '''
        write the pole figure coords for each orientation
        '''

        min_distance = 1. / ZERO_TOL

        # define point grid -1 -> +1 in x,y within unit circle
        resolution = 20.
        buffer = 1. / resolution
        grid_pts = []

        for i in range(-1 * int(resolution), int(resolution) + 1):
            for j in range(-1 * int(resolution), int(resolution) + 1):
                x = i / resolution
                y = j / resolution
                dist = sqrt(x * x + y * y)
                if dist <= (1.0 - buffer):
                    grid_pts.append([x, y])

        resolution *= pi
        for j in [(2. * pi) * (i / int(resolution)) for i in range(int(resolution) + 1)]:
            grid_pts.append([cos(j), sin(j)])

        data_array = zeros([family.shape[0] * 2 * len(self.orientDict.keys()), 2])
        for i, theta in enumerate(self.orientDict.keys()):
            cur_orient = self.orientDict[theta]
            begin = family.shape[0] * 2 * i
            end = family.shape[0] * 2 * (i + 1)
            data_array[begin:end, :] = cur_orient.getPoleFigureCoords(family, spec_axis)

        grid_array = array(grid_pts)

        # do density stuff
        # TODO: this could really use the help of a BTree or similar
        grid_pt_dens = {}
        for i, gpt in enumerate(grid_array):
            grid_pt_dens[i] = 0
            for dpt in data_array:
                diff = gpt - dpt
                if norm(diff) < binsize:
                    grid_pt_dens[i] += 1

        grid_dens_sum = 0
        for dens in grid_pt_dens.values():
            grid_dens_sum += dens

        avg_grid_pt_dens = float(grid_dens_sum) / len(grid_pts)

        for i, gpt in enumerate(grid_array):
            line = "%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n" % \
                   (gpt[0], gpt[1], 0.0, grid_pt_dens[i] / avg_grid_pt_dens, 0.0, 0.0)
            fd.write(line)

        fd.close()

    ##############################################
    def collapseToRodriguesFundamental(self):
        '''
        redefine the orientation with the symmetrically
        equivalent orientation such that the Rodrigues
        parameters fall within the fundamental zone

        if the texture has orthorhombic symmetry, then
        the fundamental zone is just the first quadrant
        of the FCC fundamental zone
        '''

        for theta in self.orientDict:
            self.orientDict[theta].inRodriguesFundamental()

    #        if self.has_ortho_symm:
    #            for theta in self.orientDict:
    #                if self.fromRodrigues:
    #                    r1,r2,r3 = self.rodDict[theta]
    #                else:
    #                    r1,r2,r3 = self.orientDict[theta].asRodrigues()
    ##############################################
    def generateTexture(self, step_size, step_num, type='2d', out='rad', conv='active'):
        '''
        type=
          '2d' will produce a set of orientations where a {111} plane normal is orthogonal to the x-z plane
          'rc' will produce a set of orientations where a <110> direction is aligned with the specimen y-axis
          'x' will produce a set of orientations where a <100> direction is aligned with the specimen x-axis
          'y' will produce a set of orientations where a <100> direction is aligned with the specimen y-axis
          'z' will produce a set of orientations where a <100> direction is aligned with the specimen z-axis
        out =
          'rad' writes euler angles in radians
          'deg' writes euler angles in degrees
        '''

        for theta in xrange(step_num):
            theta *= step_size
            self.orientDict[theta] = FCCGrain(conv, ortho=self.has_ortho_symm)
            if type == '2d':
                self.orientDict[theta].generate2DEquivalent(theta)
            elif type == 'x':
                self.orientDict[theta].xRotation(theta)
            elif type == 'y':
                self.orientDict[theta].yRotation(theta)
            elif type == 'a':
                self.orientDict[theta].aRotation(theta)
            elif type == 'rc':
                self.orientDict[theta].generateRotatedCube(theta)

    ##############################################
    def generateRandom(self, resolution=pi / 10, out='rad', conv='active'):
        '''
        generate a random texture by creating a structured grid of points
        that fills the entire Euler space
        '''

        key = 0

        for i in range(int(2 * pi / resolution)):
            phi1 = i * resolution
            for j in range(int(pi / resolution)):
                PHI = j * resolution
                for k in range(int(2 * pi / resolution)):
                    phi2 = k * resolution
                    self.orientDict[key] = FCCGrain(conv, ortho=self.has_ortho_symm)
                    eulers = array([phi1, PHI, phi2])
                    self.eulerDict[key] = eulers
                    self.orientDict[key].fromEulerAngles(array([phi1, PHI, phi2]))
                    key += 1

    ##############################################
    def generateFCCOrthoRandom(self, resolution=10., out='rad', conv='active'):
        '''
        generate a random texture by creating a structured grid of points
        that fills a single FCC-Orthotropic Euler subspace
        '''

        key = 0
        resolution = float(resolution)
        self.fromEulers = True

        for phi2 in [(pi / 2.) * (i / resolution) for i in range(int(resolution) + 1)]:
            PHI_eq1 = acos(cos(phi2) / sqrt(1. + cos(phi2) ** 2.))
            PHI_eq2 = acos(cos((pi / 2.) - phi2) / sqrt(1. + cos((pi / 2.) - phi2) ** 2.))
            for PHI in [min(PHI_eq2, PHI_eq1) * (j / resolution) for j in range(0, int(resolution) + 1, 2)]:
                for phi1 in [(pi / 2.) * (k / resolution) for k in range(int(resolution) + 1)]:
                    self.orientDict[key] = FCCGrain(conv, ortho=self.has_ortho_symm)
                    eulers = array([phi1, PHI, phi2])
                    self.eulerDict[key] = eulers
                    self.orientDict[key].fromEulerAngles(eulers)
                    key += 1

    ##############################################
    def updateByIntersection(self, other, binsize=pi / 36.):
        '''
        reduce "self" by retaining only those that are closely oriented
        with at least one in "other"
        '''

        if len(self.otherNeighbors[other].keys()) == 0:
            self.binCloselyOrientedOther(other, binsize)

        orig_keys = self.orientDict.keys()
        for key in orig_keys:
            if len(self.otherNeighbors[other][key]) == 0:
                del self.orientDict[key]

    ##############################################
    def homogenize(self, binsize=pi / 36.):
        '''
        reduce "self" by binning those closely oriented
        weighted = True will repeat ("weight") the necessary orientations to mimic
                        the original object when visualized as a pole figure
        '''

        if len(self.neighbors.keys()) != len(self.orientDict.keys()):
            self.binCloselyOrientedSelf(binsize)

        some_set = set()
        retain_set = set()

        for orient1 in self.orientDict.keys():
            if orient1 not in some_set:
                retain_set.add(orient1)
                for orient2 in self.neighbors[orient1]:
                    some_set.add(orient2)

        orig_keys = self.orientDict.keys()
        for key in orig_keys:
            if key not in retain_set:
                del self.orientDict[key]

    ##############################################
    def binCloselyOrientedSelf(self, binsize):
        '''
        build the "neighbors" dictionary of sets
        self.neighbors:
            key: Orientation key in "self"
            values: set of Orientation keys for objects in "other" that are
                oriented within the binsize misorientation
        '''

        for orient1 in self.orientDict.keys():
            self.neighbors[orient1] = set()
            self.misorientation[orient1] = []
            for orient2 in self.orientDict.keys():
                if (orient1 != orient2):
                    misorient = self.orientDict[orient1].misorientation(self.orientDict[orient2])
                    self.misorientation[orient1].append(misorient)  # gather misorientation angles
                    if misorient <= binsize:
                        self.neighbors[orient1].add(orient2)

    ##############################################
    def misorientations(self):
        '''
        build the "misorientations" dictionary of sets
        self.neighbors:
            key: Orientation key in "self"
            values: max misorientations of neighboring grains
        '''

        for orient1 in self.orientDict.keys():
            self.misorient['Grain_{}'.format(orient1 + 1)] = []
            for orient2 in self.orientDict.keys():
                if (orient1 != orient2):
                    misorient = self.orientDict[orient1].misorientation(self.orientDict[orient2])
                    self.misorient['Grain_{}'.format(orient1 + 1)].append(misorient) # gather misorientation angles
                else:
                    misorient = 0.0
                    self.misorient['Grain_{}'.format(orient1 + 1)].append(misorient)

    ##############################################
    def calc_mPrime(self):
        '''
        This function calculates the m' compatibility parameter for
        slip transmission (Luster & Morris).
        :return: None
        '''

        for orient1 in self.orientDict.keys():
            self.mp['Grain_{}'.format(orient1 + 1)] = []
            # find vecs
            n1 = self.primary_slip['Grain_{}'.format(orient1 + 1)][1]  # normal of first slip system
            d1 = self.primary_slip['Grain_{}'.format(orient1 + 1)][2] # direction of first slip (Burger's vector)
            for orient2 in self.orientDict.keys():
                # find vecs
                n2 = self.primary_slip['Grain_{}'.format(orient2 + 1)][1] # normal of second slip system
                d2 = self.primary_slip['Grain_{}'.format(orient2 + 1)][2] # direction of second slip (Burger's vector)
                if (orient1 != orient2):
                    #rotate slip normal and direction to grain orientation
                    n1 = self.orientDict[orient1]._normalize(dot(self.orientDict[orient1].R,n1))
                    d1 = self.orientDict[orient1]._normalize(dot(self.orientDict[orient1].R,d1))
                    n2 = self.orientDict[orient2]._normalize(dot(self.orientDict[orient2].R,n2))
                    d2 = self.orientDict[orient2]._normalize(dot(self.orientDict[orient2].R,d2))

                    # check orthogonality
                    assert dot(n1,d1) < 1.0E-4, 'Grain vectors not orthogonal'
                    assert dot(n2,d2) < 1.0E-4, 'Neighbor vectors not orthogonal'

                    #phi
                    uv1 = n1 / np.linalg.norm(n1)
                    uv2 = n2 / np.linalg.norm(n2)
                    cos_phi = dot(uv1, uv2)

                    #kappa
                    uv1 = d1 / np.linalg.norm(d1)
                    uv2 = d2 / np.linalg.norm(d2)
                    cos_kappa = dot(uv1, uv2)

                    mp = cos_phi * cos_kappa
                    self.mp['Grain_{}'.format(orient1 + 1)].append(mp)
                else:
                    self.mp['Grain_{}'.format(orient1 + 1)].append(np.nan)
    ##############################################
    def binCloselyOrientedOther(self, other, binsize):
        '''
        build the "otherNeighbors" dictionary of sets
        self.otherNeighbors:
            key: Orientation key in "self"
            values: set of Orientation keys for objects in "other" that are
                oriented within the binsize misorientation
        '''

        self.otherNeighbors[other] = {}
        for orient1 in self.orientDict.keys():
            self.otherNeighbors[other][orient1] = set()
            for orient2 in other.orientDict.keys():
                misorient = self.orientDict[orient1].misorientation(other.orientDict[orient2])
                if misorient <= binsize:
                    self.otherNeighbors[other][orient1].add(orient2)

    ##############################################
    def mimicRolled(self, binsize=pi / 12., resolution=20):
        '''
        generate some random orientations, then only
        keep those that are close to at lease one of the
        dominant textures for rolled components
        '''

        random = FCCTexture()
        random.generateFCCOrthoRandom(resolution=resolution)
        num_nd_rotated_cubes = 0
        num_double_planar_slip = 0

        for key in random.orientDict.keys():
            cur_orient = random.orientDict[key]
            min_distance = 1 / ZERO_TOL

            # ND rotated cube?
            for crystal_vec in FCCGrain.families[100]:
                rotated_vec = dot(cur_orient.R, crystal_vec)
                distance = fabs(FCCGrain._getAngleBetween(rotated_vec, ND))
                if distance > pi / 2: distance = pi - distance
                if distance < min_distance:
                    min_distance = distance
                    if min_distance < binsize:
                        self.addGrain(cur_orient, key)
                        num_nd_rotated_cubes += 1
                        break

            # or planar double slip?
            if min_distance > binsize:
                for crystal_vec in FCCGrain.families[111]:
                    rotated_vec = dot(cur_orient.R, crystal_vec)
                    distance = fabs(FCCGrain._getAngleBetween(rotated_vec, TD))
                    if distance > pi / 2: distance = pi - distance
                    if distance < min_distance:
                        min_distance = distance
                        if min_distance < binsize:
                            self.addGrain(cur_orient, key)
                            num_double_planar_slip += 1
                            break

        print("num_nd_rotated_cubes ", num_nd_rotated_cubes)
        print("num_double_planar_slip ", num_double_planar_slip)

    ##############################################
    def _ZhaiPaperStudy(self, n):

        conv = 'active'

        fd = open("ZhaiStudy.txt", "w")

        # select a simple orientation as "grain 1"
        self.orientDict[-1] = FCCGrain(conv, ortho=self.has_ortho_symm)
        # self.orientDict[-1].generateRotatedCube(0.0)
        self.orientDict[-1].yRotation(0.0)

        # define grain boundary normal
        GB = array([0.0, 1.0, 0.0])
        O1_traces = FCCGrain.planeTraces(self.orientDict[-1].R, GB)
        print(O1_traces)

        # define crack plane to be on first slip plane, (111)
        O1_trace = O1_traces[0, :]

        # self.wrap_generateRandom(out='deg',conv=conv)
        # self.wrap_generateTexture(2*pi/n,n,type='rc',out='deg',conv=conv)
        self.wrap_generateTexture(2 * pi / n, n, type='y', out='deg', conv=conv)

        # loop through texture and compute min alpha & misorientation
        fd.write("theta, alpha, misorientation\n")
        for cur_theta in self.orientDict:
            O2_traces = FCCGrain.planeTraces(self.orientDict[cur_theta].R, GB)
            alpha = []
            for O2_trace in O2_traces:
                tmp = fabs(FCCGrain._getAngleBetween(O1_trace, O2_trace))
                if tmp > pi / 2: tmp = pi - tmp
                alpha.append(tmp)
            s = set(alpha)
            print(len(s))
            # print O2_traces
            # print alpha
            # exit()
            fd.write("%g %g %g\n" % (cur_theta, min(alpha),
                                     self.misorientation(self.orientDict[-1],
                                                         self.orientDict[cur_theta])
                                     ))
    ##############################################
    @staticmethod
    def _normalize(v):
        # normalizes an array
        if norm(v) < 1e-10: return zeros(v.shape)
        else: return v/norm(v)
