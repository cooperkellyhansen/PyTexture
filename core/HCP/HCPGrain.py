import numpy
import numpy as np
from numpy import array,float64,dot,cross,zeros,outer
from numpy.linalg import norm

from math import *

from Orientation import *

ZERO_TOL = 1e-8
RD = array([1.,0.,0.])
ND = array([0.,0.,1.])
TD = array([0.,1.,0.])

NUM_SYSTEMS = 12 # thru pyramidal a
NUM_PLANES = 10

class HCPGrain(Orientation):

    '''
    HCPGrain is derived from the Orientation class.  In addition to the
    Orientation methods, the HCPGrain class contains methods and members
    for dealing specifically with HCP grains in an Orthorhombic specimen
    (e.g. with RD, ND, TD axes).
    '''

    systems = (
                array([1.0, 0.0, 0.0], dtype=float64),     # basal
                array([ 0.0, 1.0, 0.0], dtype=float64),
                array([ -1.0, -1.0, 0.0], dtype=float64),
                array([ 1.0 , 0.0, 0.0], dtype=float64),   # prismatic 1st order
                array([ 0.0 , 1.0, 0.0], dtype=float64),
                array([ -1.0, -1.0 , 0.0], dtype=float64),
                array([ 1.0 ,0.0, 0.0], dtype=float64),   # pyramidal (a) 1st order
                array([ 0.0, 1.0, 0.0], dtype=float64),
                array([ -1.0, -1.0, 0.0], dtype=float64),
                array([ 1.0, 1.0, 0.0], dtype=float64),
                array([ -1.0, 0.0, 0.0], dtype=float64),
                array([ 0.0, -1.0, 0.0], dtype=float64),
                )

    planes = (
              array([0., 0., 1.], dtype=float64),  # basal
              array([0., 1., 0.], dtype=float64),  # prismatic 1st order
              array([-1., 0., 0.], dtype=float64),
              array([1., -1., 0.], dtype=float64),
              array([0., -1., 1.], dtype=float64),  # pyramidal (a) 1st order
              array([1., 0.,  1.], dtype=float64),
              array([-1., 1., 1.], dtype=float64),
              array([1., -1., 1.], dtype=float64),
              array([0., 1., 1.], dtype=float64),
              array([-1., 0., 1.], dtype=float64),

              )

    ortho_symm_opers = (
                array([[1, 0, 0],[0, 1, 0],[0, 0, 1]],dtype=float64),
                array([[1, 0, 0],[0, -1, 0],[0, 0, -1]],dtype=float64),
                array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]],dtype=float64),
                array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]],dtype=float64)
                )

    symm_opers = (
                array([[1, 0, 0],[0, 1, 0],[0, 0, 1]],dtype=float64),
                array([[0.5, 0.87, 0],[-0.87, 0.5, 0],[0, 0, 1]],dtype=float64),
                array([[-0.5, 0.87, 0],[-0.87, -0.5, 0],[0, 0, 1]],dtype=float64),
                array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]],dtype=float64),
                array([[-0.5, -0.87, 0],[0.87,-0.5, 0],[0, 0, 1]],dtype=float64),
                array([[0.5, -0.87, 0],[0.87, 0.5, 0],[0, 0, 1]],dtype=float64),
                array([[1, 0, 0],[0, -1, 0],[0, 0, -1]],dtype=float64),
                array([[0.5, 0.87, 0],[0.87, -0.5, 0],[0, 0, -1]],dtype=float64),
                array([[-0.5, 0.87, 0],[0.87, 0.5, 0],[0, 0, -1]],dtype=float64),
                array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]],dtype=float64),
                array([[-0.5, -0.87, 0],[-0.87, 0.5, 0],[0, 0, -1]],dtype=float64),
                array([[0.5, -0.87, 0],[-0.87, -0.5, 0],[0, 0, -1]],dtype=float64),
                  )

    families = {
               0o01:array([[2.0, -1.0, 0.0],  # basal
                           [ -1.0, 2.0, 0.0],
                           [ -1.0, -1.0, 0.0]]),
               100:array([[ 2.0, -1.0 , 0.0],  # prismatic
                          [ -1.0, 2.0 , 0.0],
                          [-1.0, -1.0, 0.0]]),
               101:array([[0.0, -1.0, 1.0],  # pyramidal (a)
                          [ 1.0, 0.0, 1.0],
                          [-1.0, 1.0, 1.0],
                          [1.0, -1.0, 1.0],
                          [0.0, 1.0, 1.0],
                          [-1.0, 0.0, 1.0]])
               }


    #############################################
    def __init__(self,conv='passive',ortho=True):

        self.has_ortho_symm = ortho
        self.valid = True
        self.R = eye(3)
        self._conv = conv

    #############################################
    def areEquivalent(self,other,verbose=False):
        '''
        determines if two hcp grains are
        crystallographically equivalent
        '''

        if self.has_ortho_symm:
            for o_symm_oper in HCPGrain.ortho_symm_opers:
                for symm_oper in HCPGrain.symm_opers:
                    poss_equiv = dot(o_symm_oper,dot(symm_oper,other.R))
                    diff_array = self.R-poss_equiv
                    print(diff_array)
                    if ( (diff_array >= -ZERO_TOL).all() and (diff_array <= ZERO_TOL).all() ):
                        if verbose:
                            # print("R1 = ", R1)
                            print("R2 = ", poss_equiv)
                            print("when ortho_symm_oper = ", o_symm_oper)
                            print("when symm_oper = ", symm_oper)
                        return True
        else:
            for symm_oper in HCPGrain.symm_opers:
                poss_equiv = dot(symm_oper,other.R)
                diff_array = self.R-poss_equiv
                if ( (diff_array >= -ZERO_TOL).all() and (diff_array <= ZERO_TOL).all() ):
                    if verbose:
                        # print("R1 = ", R1)
                        print("R2 = ", poss_equiv)
                        # print("when ortho_symm_oper = ", o_symm_oper)
                        print("when symm_oper = ", symm_oper)
                    return True
        return False

    #############################################
    def rotateSystems(O):
        '''
        apply Orientation object rotation to:
         HCP systems (if 'active')
         Specimen basis vectors (if 'passive')
        '''

        zero_tol = 1e-10
        rot_sys = zeros([NUM_SYSTEMS,3])

        for i,sys in enumerate(HCPGrain.systems):
            sys = Orientation._normalize(sys)
            rot_sys[i,:] = Orientation._normalize(dot(O.R,sys))

        return rot_sys
    ##############################################
    def rotatePlanes(O):
        '''
        apply Orientation object rotation to HCP planes
        '''

        rot_plane = zeros([NUM_PLANES,3])

        for i,plane in enumerate(HCPGrain.planes):
            plane = Orientation._normalize(plane)
            rot_plane[i,:] = Orientation._normalize(dot(O.R,plane))

        return rot_plane
    ##############################################
    def schmidTensors(O,spec_sys=None):
        '''
        returns the list of Schmid tensors for the current object
        spec_sys is an optional input to output a specific system
          (given as the index to the "systems" member)
        '''
        rot_sys = O.rotateSystems()
        rot_planes = O.rotatePlanes()

        if spec_sys == None: all_tensors = []

        # TODO: change plane idx here
        for sys_idx,sys in enumerate(rot_sys):
            if spec_sys == None:
                if sys_idx in [0, 1, 2]: plane_idx = 0 # basal
                elif sys_idx in [3]: plane_idx = 1     # prismatic
                elif sys_idx in [4]: plane_idx = 2
                elif sys_idx in [5]: plane_idx = 3
                elif sys_idx in [6]: plane_idx = 4     #pyramdal a
                elif sys_idx in [7]: plane_idx = 5
                elif sys_idx in [8]: plane_idx = 6
                elif sys_idx in [9]: plane_idx = 7
                elif sys_idx in [10]: plane_idx = 8
                elif sys_idx in [11]: plane_idx = 9
                all_tensors.append(outer(sys,rot_planes[plane_idx]))
            else:
                if spec_sys == sys_idx:
                    all_tensors = outer(sys,rot_planes[plane_idx])

        return all_tensors
    ##############################################
    def projectSystems(O,plane_normal):
        '''
        apply Orientation object rotation to HCP systems
        project the rotated systems onto 'plane_normal'
        plane_normal = projection plane normal wrt specimen coords
        '''

        # A is vector to be projected
        # B is plane normal of projection plane

        B = Orientation._normalize(plane_normal)
        zero_tol = 1e-10

        twoD_proj = zeros([NUM_SYSTEMS,3])

        for i,sys in enumerate(HCPGrain.systems):
            sys = Orientation._normalize(sys)
            A = Orientation._normalize(dot(O.R,sys))
            AxB = Orientation._normalize(cross(A,B))
            BxAxB = Orientation._normalize(cross(B,AxB))
            normB = norm(B)
            twoD_proj[i,:] = BxAxB / (normB*normB)

            # invalid results upon normalizing the null vector
            #
            if norm(twoD_proj) > zero_tol:
                twoD_proj = Orientation._normalize(twoD_proj)

        return twoD_proj
    ##############################################
    def schmidFactors(O,load_vec):
        '''
        returns the Schmid factors (One per slip plane in structure)
        load_vec = 3-D numpy array specifying load axis
        '''

        load_vec = Orientation._normalize(load_vec)
        schmid_factors = zeros(NUM_SYSTEMS)
        normal = zeros((NUM_SYSTEMS,3))
        direction = zeros((NUM_SYSTEMS,3))

        for i,sys in enumerate(HCPGrain.systems):
            if i in [0, 1, 2]: plane_idx = 0  # basal
            elif i in [3]: plane_idx = 1  # prismatic
            elif i in [4]: plane_idx = 2
            elif i in [5]: plane_idx = 3
            elif i in [6]: plane_idx = 4  # pyramidal a
            elif i in [7]: plane_idx = 5
            elif i in [8]: plane_idx = 6
            elif i in [9]: plane_idx = 7
            elif i in [10]: plane_idx = 8
            elif i in [11]: plane_idx = 9

            sys = Orientation._normalize(sys)
            hcp_plane = HCPGrain.planes[plane_idx]

            if O._conv == 'passive':
                rot_load_vec = Orientation._normalize(dot(O.R,load_vec))

                # angle between rotated load vector and
                # crystal system direction in radians (direction)
                psi = HCPGrain._getAngleBetween(rot_load_vec,sys)

                # angle between crystal plane normal and
                # rotated load vector in radians
                xi = HCPGrain._getAngleBetween(rot_load_vec,hcp_plane)

            elif O._conv == 'active':
                rot_sys = Orientation._normalize(dot(O.R,sys))
                rot_hcp_plane = Orientation._normalize(dot(O.R,hcp_plane))

                # angle between rotated slip system and principle
                # loading direction in radians
                psi = HCPGrain._getAngleBetween(rot_sys,load_vec)

                # angle between rotated slip plane normal and principle
                # loading direction in radians
                xi = HCPGrain._getAngleBetween(rot_hcp_plane,load_vec)

            schmid_factors[i] = fabs(cos(psi) * cos(xi))
            normal[i] = hcp_plane # normals of planes
            direction[i] = sys # directions of slip

        return schmid_factors,normal,direction
    ##############################################
    def maxSchmidFactor(O,load_vec):
        '''
        returns the maximum Schmid factor
        load_vec = 3-D numpy array specifying load axis
        '''
        basal_plane = array([0.,0.,1.],dtype=float64)

        m,n,d = O.schmidFactors(load_vec)
        max_schmid = max(m)
        max_plane = n[np.argmax(np.array(m))]
        max_direction = d[np.argmax(np.array(m))]

        return max_schmid,max_plane,max_direction
    ##############################################
    def planeTraces(O,specimen_normal):
        '''
        returns the traces that the planes make on
        a specified 'specimen_normal'
        specimen_normal = 3-D numpy array
        '''

        traces=zeros([4,3])
        for plane_id,plane in enumerate(HCPGrain.planes):
            nplane = Orientation._normalize(plane)
            rot_plane = Orientation._normalize(dot(O.R,nplane))
            spec_plane = Orientation._normalize(specimen_normal)
            traces[plane_id,:] = Orientation._normalize(cross(spec_plane,rot_plane))

        return traces

    ##############################################
    def getPoleFigureCoords(self,family,spec_axis):
        '''
        returns a numpy array of x,y pole figure points for
        the given family and specimen axis
        family = HCPGrain.families[key]
        spec_axis = RD,TD,ND
        '''

        if not self.has_ortho_symm:
            print("error: getting PF coords only works with Ortho symmetry!")
            return None

        data = zeros([family.shape[0]*2,2])
        row = 0

        for crystal_vec in family:
            # do both +tive and -tive of the direction
            for i in range(2):
                if i==1: crystal_vec *= -1.0

                if self._conv == 'passive':
                    r,t,n = HCPGrain._normalize(dot(self.R.T,crystal_vec))
                elif self._conv == 'active':
                    r,t,n = HCPGrain._normalize(dot(self.R,crystal_vec))
                else: return None

                if (spec_axis - ND).all() == 0:
                    if n < 0.0: n *= -1.0
                    d = n/(1.0+n)
                    x = r * (1.0-d)
                    y = t * (1.0-d)

                elif (spec_axis - RD).all() == 0:
                    if r < 0.0: r *= -1.0
                    d = r/(1.0+r)
                    x = t * (1.0-d)
                    y = n * (1.0-d)
                elif (spec_axis - TD).all() == 0:
                    if t < 0.0: t *= -1.0
                    d = t/(1.0+t)
                    x = n * (1.0-d)
                    y = r * (1.0-d)
                else:
                    print("error, no such specimen axis!")
                    return None

                data[row,:] = [x,y]
                row += 1

        return data
    ##############################################
    def getInvPoleFigureCoords(self,spec_axis):
        '''
        returns a numpy array of x,y pole figure points for
        the given specimen axis
        spec_axis = "RD","TD","ND"
        '''

        valid_coord = []

        pt_0 = [0.0, 0.0]
        pt_1 = [0.36602540378458909, 0.36602540378458909]
        pt_2 = [tan(pi/8), 0.0]
        total_area = 0.078786686

        for symm_oper in HCPGrain.symm_opers:
            poss_equiv = dot(symm_oper,self.R)
            for family in HCPGrain.families.keys():
                coords = self.getPoleFigureCoords(HCPGrain.families[family],spec_axis)

        #TODO: Not done yet!
    ##############################################
    def equivEulerAngles(O):
        #TODO: I dont think this is going to work with the HCP structure
        '''
        returns the 12 crystallographically
        equivalent Euler angles
        '''

        orig_R = O.R
        norm_min_r = 1./ZERO_TOL

        equiv_eulers = zeros([NUM_SYSTEMS,3])

        for i, symm_oper in enumerate(HCPGrain.symm_opers):
            O.R = dot(symm_oper,orig_R)
            print(O.asEulerAngles())
            equiv_eulers[i,:] = O.asEulerAngles()

        O.R = orig_R

        return equiv_eulers
    ##############################################
    def inRodriguesFundamental(O):
        '''
        updates the object to the symmetrical equivalent
        that is in the Rodrigues fundamental zone
        '''

        orig_R = O.R
        norm_min_r = 1./ZERO_TOL

        for symm_oper in HCPGrain.symm_opers:
            O.R = dot(symm_oper,orig_R)
            r = O.asRodrigues()
            norm_r = norm(r)
            if norm_r < norm_min_r:
                norm_min_r = norm_r
                min_R = O.R


        # TODO: not sure this statement is true for HCP
        # for HCP with ortho symm, there are 4 equivalent sets
        # of Rodrigues parameters. Those returned here will
        # always correspond to the first quadrant.
        #if O.has_ortho_symm:
         #   for o_symm_oper in HCPGrain.ortho_symm_opers:
          #      o_symm_equiv = dot(o_symm_oper,orig_R)
           #     O.R = o_symm_equiv
            #    r = O.asRodrigues()
             #   if r[0]>0.0 and r[1]>0.0:
              #      return None

        #else:

        O.R = min_R
        return None
    ##############################################
    def inHCPOrthoSubspace(O):
        # TODO: I dont think this is going to work with the HCP structure
        '''
        returns the set of Euler angles
        that are symmetrically equivalent and in the
        reduced HCP-Ortho subspace of Euler space
        '''

        orig_R = O.R
        for symm_oper in HCPGrain.symm_opers:
            O.R = dot(symm_oper,orig_R)
            eulers,in_subspace = O.checkHCPOrthoSubspace()
            if in_subspace:
                return eulers

        O.R = orig_R

        exit(1)
        return None
    ##############################################
    def checkHCPOrthoSubspace(O):
        '''
        checks if euler angles are in HCP-Ortho subspace
          - if PHI=0.0, then a simple check is done to see if
            one of the infinitely many descriptions (see asEulerAngles)
            is in the subspace.
        '''

        check=False
        eulers = O.asEulerAngles()
        pad = 0.0001
        bound = pi/2.+ pad

        if Orientation._fequals(eulers[1], 0.0) and \
            Orientation._fequals(eulers[2], 0.0):
            if eulers[0] <= bound:
                check=True
                return eulers,check
            elif eulers[0]/2.0 <= bound:
                check=True
                eulers[0] /= 2.0
                eulers[2] /= eulers[0]
                return eulers,check
        elif eulers[2] <= bound:
            PHI_eq1 = acos(cos(eulers[2])/sqrt(1.+cos(eulers[2])**2.))
            PHI_eq2 = acos(cos((pi/2.)-eulers[2])/sqrt(1.+cos((pi/2.)-eulers[2])**2.))
            if eulers[1] < min(PHI_eq1,PHI_eq2) + pad:
                if eulers[0] <= bound:
                    check=True
            return eulers,check
        else:
            return eulers,check
    ##############################################
    def misorientation(self,other):
        '''
        returns the minimum misorientation between objects.
        Goes through all symmetrically equivalent orientations
        and picks minimum.
        '''

        if self.checkValid and other.checkValid:

            min_misorientation = 10000

            for symm_oper in HCPGrain.symm_opers:
                if self._conv == other._conv:
                    dir_cos = 0.5*(trace(dot(symm_oper,dot(self.R,other.R.T)))-1.0)
                else:
                    dir_cos = 0.5*(trace(dot(symm_oper,dot(self.R,other.R)))-1.0)
                try:
                    # this is to catch values that are just barely
                    # above +1 or barely below -1 and handle the
                    # the domain error appropriately
                    if Orientation._fequals(dir_cos, 1.0):
                        misorientation = 0.0
                    if Orientation._fequals(dir_cos, -1.0):
                        misorientation = pi
                    else:
                        misorientation = acos(dir_cos)
                except ValueError:
                    pass
                    #print("direction cosine = ", dir_cos)
                    #print("acos() not defined for this direction cos!")

                if misorientation < min_misorientation:
                    min_misorientation = misorientation

            return min_misorientation

        else:
            return None

    ##############################################
    def getAlignment(self,family,spec_axis):
        '''
        returns the minimum angle between all possibilities
        of the specified HCP family and a specimen axis
        family = HCPGrain.families[key]
        spec_axis = numpy array
        '''

        min_distance = 1.0/ZERO_TOL
        min_vec = None

        for crystal_vec in family:
            rotated_vec = dot(self.R,crystal_vec)
            distance = fabs(HCPGrain._getAngleBetween(rotated_vec,spec_axis))
            if distance > pi/2:
                distance = pi-distance
            if distance < min_distance:
                min_distance = distance
                min_vec = crystal_vec

        # Used to return the angle but switched to vector.
        return min_vec
    ##############################################
    # TODO: I don't think these next 3-4 functions work. Need to adapt to HCP.
    def generateXZDoublePlanarSlip(self,theta):
        '''
        will produce an Orientation object where the (111)
        plane normal is aligned with the specimen y-axis
        '''

        self.zRotation(pi/4.0)
        R1 = self.R

        x_angle = -atan(sqrt(2.0)/2.0)
        self.xRotation(x_angle)
        R2 = self.R

        self.yRotation(theta)
        R3 = self.R

        if self._conv == 'active':
            self.R = dot(R3,dot(R2,R1))

        elif self._conv == 'passive':
            self.R = dot(R3,dot(R2,R1)).T

        else:
            print("No such convention")
    ##############################################
    def generateNDRotatedCube(self,theta):
        '''
        will produce an Orientation object where the [110]
        direction is aligned with the specimen ND axis
        '''

        self.zRotation(pi/4.0)
        R1 = self.R

        self.xRotation(pi/2.0)
        R2 = self.R

        self.zRotation(theta)
        R3 = self.R

        if self._conv == 'active':
            self.R = dot(R3,dot(R2,R1))

        elif self._conv == 'passive':
            self.R = dot(R3,dot(R2,R1)).T

        else:
            print("No such convention")
    ##############################################
    def generateRDRotatedCube(self,theta):
        '''
        will produce an Orientation object where the [110]
        direction is aligned with the specimen RD axis
        '''

        self.zRotation(-pi/4.0)
        R1 = self.R

        self.xRotation(theta)
        R2 = self.R

        if self._conv == 'active':
            self.R = dot(R2,R1)

        elif self._conv == 'passive':
            self.R = dot(R2,R1).T

        else:
            print("No such convention")
    ##############################################
    def generateTDRotatedCube(self,theta):
        '''
        will produce an Orientation object where the [110]
        direction is aligned with the specimen TD axis
        '''

        self.zRotation(pi/4.0)
        R1 = self.R

        self.yRotation(theta)
        R2 = self.R

        if self._conv == 'active':
            self.R = dot(R2,R1)

        elif self._conv == 'passive':
            self.R = dot(R2,R1).T

        else:
            print("No such convention")
    ##############################################
    @staticmethod
    def _getAngleBetween(a,b):
        '''
        return the angle between two systems
        '''

        a = Orientation._normalize(a)
        b = Orientation._normalize(b)
        dotab = dot(a,b)
        try:
            if Orientation._fequals(fabs(dotab), 1.0): return 0.0
            else: return acos(dotab)
        except ValueError:
            print("vec a = ", a)
            print("vec b = ", b)
            print("dot(a,b) = ", dotab)\
            ("acos(%g) is not defined!" % dotab)