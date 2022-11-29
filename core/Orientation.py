from functools import reduce

import numpy
from numpy import array,dot,float64,any,trace,cross,zeros,eye,vectorize
from numpy.linalg import inv,norm,det

from fractions import Fraction

import math
from math import *

ZERO_TOL = 1e-8
R2D = 180.0/pi

class Orientation:
    '''
        This class contains methods for converting among various
        descriptions of an orientation: Rotation Matrix, Euler Angles,
        Quaternions, Rodrigues parameters, Angle-Axis pair, and
        Miller indices.

        Every Orientation object is instantiated with one of the above
        descriptions using a from* method. The Rotation Matrix will be
        built from the input. Then the as* methods are used to translate
        the Rotation Matrix into the desired description.

        The default convention of the Rotation Matrix used here is in the
        'active' sense: in Materials Science vocabulary, this means
        moving the crystal with respect to the specimen. To translate
        between active and passive, simply use the switchConvention()
        method. Also, a class object can be initialized by passing the
        optional parameter, conv='passive'. It might be convenient to
        get the 'passive' description of some available parameter from
        the 'active' orientation matrix (or vice-versa). In this case,
        one can redefine the '._conv' member and then use the 'as*'
        methods as usual.
    '''

    _valid_convs = ['active','passive']

    def __init__(self,conv='passive'):

        self.valid = True
        self.R = eye(3)
        self._conv = conv
    ##############################################
    def checkValid(self):
        '''
        this method does a few checks to ensure
        that the orientation matrix is valid
        '''

        self.valid = False

        try:
            assert abs(norm(self.R.T - inv(self.R))) <= ZERO_TOL
            self.valid = True
        except AssertionError:
            print("this matrix is not a rotation matrix, transpose(R) != inverse(R)!")
            print(norm(self.R.T - inv(self.R)))
            self.valid = False

        try:
            assert abs(det(self.R)-1.0) <= (1.0+ZERO_TOL)
            self.valid = True
        except AssertionError:
            print("this rotation matrix is not a rotation matrix, det(R) != 1!")
            print(det(self.R))
            self.valid = False

        if self._conv not in Orientation._valid_convs:
            self.valid = False

        return None
    ##############################################
    def switchConvention(self):
        '''
        this method allows switching between active
        and passive transformations.
        - self._conv will be switched and
        - self.R = self.R.T
        '''

        self.checkValid()
        self.R = self.R.T

        if self._conv == 'active': self._conv = 'passive'
        else: self._conv = 'active'
        return None
    ##############################################
    def xRotation(self,theta):
        '''
        do a rotation about the x axis
        'active' - will rotate the point-space w.r.t. fixed basis vectors
        'passive' - will rotate the basis vectors w.r.t. fixed point-space
        '''

        c = cos(theta)
        s = sin(theta)

        if self._conv == 'active':
            self.R = array([
                        [1,0,0],
                        [0, c, -s],
                        [0, s, c]
                        ])

        elif self._conv == 'passive':
            self.R = array([
                        [1,0,0],
                        [0, c, s],
                        [0, -s, c]
                        ])
        else:
            print("No such convention")

        return None
    ##############################################
    def yRotation(self,theta):
        '''
        do a rotation about the y axis
        'active' - will rotate the point-space w.r.t. fixed basis vectors
        'passive' - will rotate the basis vectors w.r.t. fixed point-space
        '''

        c = cos(theta)
        s = sin(theta)

        if self._conv == 'active':
            self.R = array([
                    [c, 0, -s],
                    [0, 1, 0],
                    [s, 0, c]
                    ])

        elif self._conv == 'passive':
            self.R = array([
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]
                    ])
        else:
            print("No such convention")

        return None
    ##############################################
    def zRotation(self,theta):
        '''
        do a rotation about the z axis
        'active' - will rotate the point-space w.r.t. fixed basis vectors
        'passive' - will rotate the basis vectors w.r.t. fixed point-space
        '''

        c = cos(theta)
        s = sin(theta)

        if self._conv == 'active':
            self.R = array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0,0,1]
                        ])

        elif self._conv == 'passive':
            self.R = array([
                        [c, s, 0],
                        [-s, c, 0],
                        [0,0,1]
                        ])
        else:
            print("No such convention")

        return None

    ##############################################
    def applyZRotation(self,theta):
        '''
        apply rotation about z to object
        'active' - will rotate the point-space w.r.t. fixed basis vectors
        'passive' - will rotate the basis vectors w.r.t. fixed point-space
        '''


        c = cos(theta)
        s = sin(theta)

        if self._conv == 'active':
            zRot = array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0,0,1]
                        ])

        elif self._conv == 'passive':
            zRot = array([
                        [c, s, 0],
                        [-s, c, 0],
                        [0,0,1]
                        ])

        self.R = dot(zRot,self.R)

    ##############################################
    def fromRotationMatrix(self,Rot):
        '''
        build the Orientation object from a rotation matrix
        '''
        self.R = Rot
        self.checkValid()
    ##############################################
    def fromEulerAngles(self,eulers):
        '''
        build the Orientation object from Bunge-Euler angles
        '''

        c_phi1 = cos(eulers[0])
        s_phi1 = sin(eulers[0])
        c_PHI = cos(eulers[1])
        s_PHI = sin(eulers[1])
        c_phi2 = cos(eulers[2])
        s_phi2 = sin(eulers[2])

        Rot = zeros([3,3])

        if self._conv == 'active':
            Rot[0][0] =  c_phi1*c_phi2 - s_phi1*s_phi2*c_PHI
            Rot[0][1] = -c_phi1*s_phi2 - s_phi1*c_phi2*c_PHI
            Rot[0][2] =  s_phi1*s_PHI
            Rot[1][0] =  s_phi1*c_phi2 + c_phi1*s_phi2*c_PHI
            Rot[1][1] = -s_phi1*s_phi2 + c_phi1*c_phi2*c_PHI
            Rot[1][2] = -c_phi1*s_PHI
            Rot[2][0] =  s_phi2*s_PHI
            Rot[2][1] =  c_phi2*s_PHI
            Rot[2][2] =  c_PHI

        elif self._conv == 'passive':
            Rot[0][0] =  c_phi1*c_phi2 - s_phi1*s_phi2*c_PHI
            Rot[0][1] =  s_phi1*c_phi2 + c_phi1*s_phi2*c_PHI
            Rot[0][2] =  s_phi2*s_PHI
            Rot[1][0] = -c_phi1*s_phi2 - s_phi1*c_phi2*c_PHI
            Rot[1][1] = -s_phi1*s_phi2 + c_phi1*c_phi2*c_PHI
            Rot[1][2] =  c_phi2*s_PHI
            Rot[2][0] =  s_phi1*s_PHI
            Rot[2][1] = -c_phi1*s_PHI
            Rot[2][2] =  c_PHI

        else:
            print("No such convention")

        self.R = Rot

        return None
    ##############################################
    def fromAngleAxis(self,theta,vec):
        '''
        build the Orientation object from angle-axis
        '''

        vec = self._normalize(vec)
        vx,vy,vz = vec

        c = cos(theta)
        s = sin(theta)
        t = 1-cos(theta)

        if self._conv == 'active':
            self.R = array([
                       [vx**2.0*t+c,vx*vy*t-vz*s,vx*vz*t+vy*s],
                       [vy*vx*t+vz*s,vy**2.0*t+c,vy*vz*t-vx*s],
                       [vz*vx*t-vy*s,vz*vy*t+vx*s,vz**2.0*t+c]
                       ])
        elif self._conv == 'passive':
             self.R = array([
                       [vx**2.0*t+c,vy*vx*t+vz*s,vz*vx*t-vy*s],
                       [vx*vy*t-vz*s,vy**2.0*t+c,vz*vy*t+vx*s],
                       [vx*vz*t+vy*s,vy*vz*t-vx*s,vz**2.0*t+c]
                       ])
        else:
            print("No such convention")

        return None
    ##############################################
    def fromQuaternion(self,quat):
        '''
        build the Orientation object from quaternions
        '''

        rod = quat[0:3]/quat[-1]
        self.fromRodrigues(rod)

        return None
    ##############################################
    def fromRodrigues(self,rod):
        '''
        build the Orientation object from Rodrigues parameters
        '''

        mag = norm(rod) ;
        ang = 2.0 * atan(mag) ;

        if mag < ZERO_TOL: n = 0.0
        else: n = rod/mag ;

        c = cos(ang)
        s = sin(ang)
        t = (1.0-c)

        if self._conv == 'active':
            if not self._fequals(ang,0.0):
                self.R[0][0] = (1.0-n[0]*n[0])*c + n[0]*n[0]
                self.R[1][0] = n[0]*n[1]*t + n[2]*s
                self.R[2][0] = n[0]*n[2]*t - n[1]*s

                self.R[0][1] = n[1]*n[0]*t - n[2]*s
                self.R[1][1] = (1-n[1]*n[1])*c + n[1]*n[1]
                self.R[2][1] = n[1]*n[2]*t + n[0]*s

                self.R[0][2] = n[2]*n[0]*t + n[1]*s
                self.R[1][2] = n[2]*n[1]*t - n[0]*s
                self.R[2][2] = (1.0-n[2]*n[2])*c + n[2]*n[2]

        elif self._conv == 'passive':
            if not self._fequals(ang,0.0):
                self.R[0][0] = (1.0-n[0]*n[0])*c + n[0]*n[0]
                self.R[0][1] = n[0]*n[1]*t + n[2]*s
                self.R[0][2] = n[0]*n[2]*t - n[1]*s

                self.R[1][0] = n[1]*n[0]*t - n[2]*s
                self.R[1][1] = (1-n[1]*n[1])*c + n[1]*n[1]
                self.R[1][2] = n[1]*n[2]*t + n[0]*s

                self.R[2][0] = n[2]*n[0]*t + n[1]*s
                self.R[2][1] = n[2]*n[1]*t - n[0]*s
                self.R[2][2] = (1.0-n[2]*n[2])*c + n[2]*n[2]
        else:
            print("No such convention")

        return None
    ##############################################
    def fromMillerIndices(self,plane,dir):
        '''
        build the Orientation object from Miller indices
        '''

        h = plane[0]
        k = plane[1]
        l = plane[2]

        u = dir[0]
        v = dir[1]
        w = dir[2]

        pn = norm(plane)
        dn = norm(dir)
        pndn = pn*dn

        if self._conv == 'active':
            self.R = array([
                            [u/dn,v/dn,w/dn],
                            [(k*w-l*v)/pndn,(l*u-h*w)/pndn,(h*v-k*u)/pndn],
                            [h/pn,k/pn,l/pn]
                            ])
        elif self._conv == 'passive':
            self.R = array([
                            [u/dn,(k*w-l*v)/pndn,h/pn],
                            [v/dn,(l*u-h*w)/pndn,k/pn],
                            [w/dn,(h*v-k*u)/pndn,l/pn]
                            ])
        else:
            print("No such convention")

        return None
    ##############################################
    def asEulerAngles(self):
        '''
        This method will return the Bunge-Euler description of the
        Orientation object. Each Euler angle will be expressed in the
        (phi1,PHI,phi2) in (0->2pi,0->pi,0->2pi) domain

        When PHI = 0.0 there are an infinite number of valid descriptions
        that could be returned, e.g. (phi1,PHI,phi2) = (pi,0.0,0.0) is
        the same exact orientation as (pi/2,0.0,pi/2) and (0.0,0.0,pi), etc.
        In this case, phi2 is set equal to zero to return a specific
        description.
        '''

        self.checkValid()

        if self.valid:

            # PHI
            PHI = acos(self.R[2][2])

            # quick check
            try: assert (PHI <= (pi+ZERO_TOL) and PHI >= (0.0-ZERO_TOL))
            except AssertionError: print("illegal PHI value")

            if self._conv == 'active':
                if ( (not (Orientation._fequals(self.R[2][0],0.0) and Orientation._fequals(self.R[2][1],0.0)) or \
                    not (Orientation._fequals(self.R[0][2],0.0) and Orientation._fequals(self.R[1][2],0.0))) and \
                    not Orientation._fequals(cos(PHI),0.0) ):

                    phi1 = atan2(self.R[0][2],-self.R[1][2])
                    phi2 = atan2(self.R[2][0],self.R[2][1])

                # this 'else' is almost never activated for measured orientations
                # but is often called in simulated orientations
                else:
                    if Orientation._fequals(self.R[2][2],1.0):
                        a = ( self.R[0][0] + self.R[1][1] ) / (1.0 + self.R[2][2])
                        b = ( self.R[1][0] - self.R[0][1] ) / (1.0 + self.R[2][2])
                    else:
                        a = ( self.R[0][1] + self.R[1][0] ) / (self.R[2][2] - 1.0)
                        b = ( self.R[1][1] - self.R[0][0] ) / (self.R[2][2] - 1.0)

                    phi1 = atan2(b,a)
                    phi2 = 0.0

            else:
                if ( not (Orientation._fequals(self.R[2][0],0.0) and Orientation._fequals(self.R[2][1],0.0)) or \
                    not (Orientation._fequals(self.R[0][2],0.0) and Orientation._fequals(self.R[1][2],0.0)) ):

                    phi1 = atan2(self.R[2][0],-self.R[2][1])
                    phi2 = atan2(self.R[0][2],self.R[1][2])

                else:
                    if Orientation._fequals(self.R[2][2],1.0):
                        a = ( self.R[0][0] + self.R[1][1] ) / (1.0 + self.R[2][2])
                        b = ( self.R[0][1] - self.R[1][0] ) / (1.0 + self.R[2][2])
                    else:
                        a = ( self.R[0][1] + self.R[1][0] ) / (self.R[2][2] - 1.0)
                        b = ( self.R[1][1] - self.R[0][0] ) / (self.R[2][2] - 1.0)

                    phi1 = atan2(b,a)
                    phi2 = 0.0

            if phi1 < 0.0: phi1 += 2.0*pi
            if phi2 < 0.0: phi2 += 2.0*pi

            if Orientation._fequals(phi1, 2.*pi): phi1=0.0
            if Orientation._fequals(PHI, 2.*pi): PHI=0.0
            if Orientation._fequals(phi2, 2.*pi): phi2=0.0

            return array([phi1,PHI,phi2])

        else:
            return array([None,None,None])
    ##############################################
    def asAngleAxis(self):
        '''
        return the angle-axis description of the Orientation object
        '''

        self.checkValid()
        arg = (trace(self.R)-1.0)/2.0

        if Orientation._fequals(fabs(arg), 1.0): theta = 0.0
        else: theta = acos(arg)

        vec=zeros(3)

        if self._conv == 'active':
            vec[0]= self.R[2][1] - self.R[1][2]
            vec[1]= self.R[0][2] - self.R[2][0]
            vec[2]= self.R[1][0] - self.R[0][1]

        else:
            vec[0]= self.R[1][2] - self.R[2][1]
            vec[1]= self.R[2][0] - self.R[0][2]
            vec[2]= self.R[0][1] - self.R[1][0]


        return theta,self._normalize(vec)
    ##############################################
    def asQuaternion(self):
        '''
        return the quaternion description of the Orientation object
        '''

        self.checkValid()

        rod = self.asRodrigues()
        theta_2 = atan(norm(rod))
        cos_theta_2 = cos(theta_2)

        return cos_theta_2 * array([1, rod[0], rod[1], rod[2]])
    ##############################################
    def asRodrigues(self):
        '''
        return the Rodrigues parameters description of the Orientation object
        '''

        self.checkValid()

        theta,vec = self.asAngleAxis()
        t = tan(theta/2.0)
        rod = t*vec
        return rod
    ##############################################
    def asMillerIndices(self):
        '''
        return the Miller indices description of the Orientation object
        '''

        self.checkValid()

        l = self.R[2][2]

        if self._conv == 'active':
            h = self.R[2][0]
            k = self.R[2][1]
            u = self.R[0][0]
            v = self.R[0][1]
            w = self.R[0][2]
        else:
            h = self.R[0][2]
            k = self.R[1][2]
            u = self.R[0][0]
            v = self.R[1][0]
            w = self.R[2][0]

        miller_plane = array([h,k,l])
        miller_dir = array([u,v,w])

        try:
            return Orientation._rationalize_millers(miller_plane),Orientation._rationalize_millers(miller_dir)
        except ValueError:
            return miller_plane, miller_dir

        return None

    ##############################################
    @staticmethod
    def _rationalize_millers(millers):
        # return millers divided by greatest common denominator
        max_miller = max(millers)
        millers /= max_miller

        if max_miller < 0.0:
            millers *= -1

        h_fract = Fraction(str(millers[0])).limit_denominator(13)
        k_fract = Fraction(str(millers[1])).limit_denominator(13)
        l_fract = Fraction(str(millers[2])).limit_denominator(13)

        h_fract_denom = h_fract.denominator
        k_fract_denom = k_fract.denominator
        l_fract_denom = l_fract.denominator

        denoms = array([h_fract_denom,k_fract_denom,l_fract_denom])
        lcm = Orientation._lcmm(denoms)
        millers *= lcm

        vecfunc = vectorize(Orientation._round)

        return vecfunc(millers)
    ##############################################
    @staticmethod
    def _round(a):
        # rounds to the nearest
        return round(a)
    ##############################################
    @staticmethod
    def _normalize(v):
        # normalizes an array
        if norm(v) < 1e-10: return zeros(v.shape)
        else: return v/norm(v)
    ##############################################
    @staticmethod
    def _fequals(val1,val2):
        if ( abs(val1-val2) <= ZERO_TOL ):
            return True
        else:
            return False
    ##############################################
    @staticmethod
    def _gcd(a, b):
        # will return greatest common divisor
        # using Euclid's Algorithm
        while b:
            a, b = b, a % b
        return a
    ##############################################
    @staticmethod
    def _lcm(a, b):
        # will return lowest common multiple
        return a * b / Orientation._gcd(a, b)
    ##############################################
    @staticmethod
    def _lcmm(v):
        # return lcm of v
        return reduce(Orientation._lcm, v)
    ##############################################
    @staticmethod
    def test_it():
        # actively rotate point about y-axis by 45 degrees
        #
        O = Orientation('active')
        p=array([1,0,0],dtype=float64)
        p = O._normalize(p)
        O.zRotation(pi/4)
        rotated_p = O._normalize(dot(O.R,p))
        print("Actively rotated crystal axes wrt specimen z-axis by 45 degrees...")
        print("Components of crystal axis [1,0,0] wrt to specimen axes is now:")
        print(rotated_p)

        # passively rotate point about y-axis by 45 degrees
        #
        O = Orientation('passive')
        p=array([1,0,0],dtype=float64)
        p = O._normalize(p)
        O.zRotation(pi/4)
        rotated_p = O._normalize(dot(O.R,p))
        print("Passively rotated specimen axes wrt specimen z-axis by 45 degrees...")
        print("Components of specimen axis [1,0,0] wrt to crystal axes is now:")
        print(rotated_p)

        # test euler angles
        #
        phi1= 0.0
        PHI= pi/4
        phi2= pi
        eulers = array([phi1,PHI,phi2],dtype=float64)
        O = Orientation('active')
        O.fromEulerAngles(eulers)
        print("These two entries should be equal")
        print(eulers, O.asEulerAngles())

        # verification with Randle & Engler, Table 2.1 (1st entry)
        #
        print("Randle & Engler verification test")
        d2r = math.pi/180.0
        eulers = array([301.0,36.7,26.6],dtype=float64)*d2r
        O = Orientation(conv='passive')
        O.fromEulerAngles(eulers)
        print(O.asRodrigues())
        print(O.asQuaternion())
        print(O.asAngleAxis())
        print(O.asEulerAngles())
        print(O.R)
