# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 07:43:03 2016

@author:    Matthias Sachs (email: msachs@math.duke.edu) 
            Anton Martinsson (email: Anton.Martinsson@ed.ac.uk)

Copyright: Duke University & Univeristy of Edinburgh

Please contact the authors if you would like to reuse the code outside 
of the tutorial session


"""

import numpy as np
import abc



class Model(object):
    '''
    Base class for force fields. Each derived class must implement 
    the functions com_force and comp_potential.
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, dim):
        '''
        For every model the number of dimensions need to be specified
        '''
        self.dim = dim
        
    @abc.abstractmethod
    def comp_force(self, q):
        '''
        returns the force for the provided position vector  
        
        param: q: position vector, numpy array of length self.dim
            
        '''
        raise NotImplementedError()
        
    @abc.abstractmethod
    def comp_potential(self, q):
        '''
        returns the potential energy for the provided position vector  
        
        param: q: position vector, numpy array of length self.dim
            
        '''
        raise NotImplementedError()

    def apply_boundary_conditions(self, q):
        '''
        Function which updates the position variable q according to the 
        boundary condition of the positional domain 
        '''
        pass

class HarmonicOscillator(Model):       
        
    def __init__(self, dim=1, k=1.0, rp=0.0):
        
        super(HarmonicOscillator,self).__init__(dim)
        
        self.rp_vec = rp * np.ones(self.dim)
        self.k_vec = k * np.ones(self.dim)
    
        

    def comp_force(self, x):
        return - self.k_vec * (x - self.rp_vec)
        
    def comp_potential(self, x):
        return .5 * np.sum( self.k_vec * (x - self.rp_vec)**2,axis=0)

class MultiScaleModel(Model):
    __metaclass__ = abc.ABCMeta
    
    def comp_fastForce(self, q):
        raise NotImplementedError()
    def comp_slowForce(self, q):
        raise NotImplementedError()
        
    def comp_force(self, q):
        return self.comp_slowForce(q) + self.comp_fastForce(q)
    
    
    def comp_fastPotential(self, q):
        raise NotImplementedError()    
    def comp_slowPotential(self, q):
        raise NotImplementedError()
        
    def comp_potential(self, q):
        return self.comp_slowPotential(q) + self.comp_fastPotential(q)
    
class DoublePendulum(MultiScaleModel):
    '''
    Implements a simple double pendulum in the plane consisting of two particles
    q1 = (q_1x,q_1y), and q2 = (q_2x,q_2y), which interact with a harmonic spring
    of rest length r12 and stiffnes constant k12, and the first particle q1 feels fhe force of a harmonic
    spring connected with the originin of rest length r1 and stiffnes constant k1 
    
    It is assumed that there is a scale separation between the degrees of freedom
    describing the relative movement of the particles and the degrees of freedom
    describing the movement of the first particle relative to the origin, i.e.,
    the stiffness constants k1 and k12 are typically chosen such that k12 << k1.
    '''
    
    def __init__(self, k1=1.0, r1=1.0, k12=1.0, r12=1.0):
        '''    
        model.q[:2] correspond to the x and y coordinates of the first particle, repecicely 
        model.q[2:] correspond to the x and y coordinates of the first particle, repecicely 
        '''
        super(DoublePendulum, self).__init__(dim=4)
        self.k1 = k1
        self.k12 = k12
        self.r1 = r1
        self.r12 = r12
    
    def comp_fastForce(self, q):
        force = np.zeros(self.dim)
        dist1 = np.linalg.norm(q[:2])
        force[:2] = - q[:2] * self.k1 * (dist1-self.r1)/dist1 
        return force
    
    def comp_slowForce(self, q):
        force = np.zeros(self.dim)
        dist12 = np.linalg.norm(q[:2] - q[2:])
        force[2:] = (q[:2]-q[2:]) * self.k12  * (dist12-self.r12)/dist12
        force[:2] = - force[2:]
        return force
    
    def comp_fastPotential(self, q):
        return  .5 * self.k1* (np.linalg.norm(q[:2]) - self.r1)**2

    def comp_slowPotential(self, q):
        return .5 * self.k12*(np.linalg.norm(q[:2] - q[2:])- self.r12)**2       
      
    def comp_potential(self, q):
        a = .5 * self.k1* (np.linalg.norm(q[:2]) - self.r1)**2 
        b = .5 * self.k12*(np.linalg.norm(q[:2] - q[2:])- self.r12)**2
        return  a+b
    
    
class CosineModel(Model):
    """ 
    Class which implements the force and potential for the cosine model
    """
    def __init__(self, dim=1, L=2.*np.pi):
        """ 
        Init function for the class
        :param L: length of periodic box
        """
        super(CosineModel, self).__init__(dim)
        # Length of the periodic box
        self.L = L
    
    def comp_force(self,q):
        """ updates the force internally from the current position
        """
        return np.sin(q)
        
    def comp_potential(self, q):
        """ returns the potential at the current position
        """
        return np.cos(q)

    def apply_boundary_conditions(self, q):
        q = np.mod(q, self.L)

class CubicDoubleWell(Model):
    """ 
    Class implementing the force and potential for a double well potential of
    the form U(q) = (b - a/2) * (q^2-1)^2 + a/2 * (q+1)
    """
    def __init__(self, dim=1, a=1.0,b=1.0):
    
        super(CubicDoubleWell, self).__init__(dim)
        self.a = a
        self.b = b
        
    def comp_force(self, q):
        return - 4.0*(self.b-.5 * self.a)*q*(q**2-1.0) - .5 * self.a
        
    def comp_potential(self, q):
        return (self.b -.5 * self.a)*(q**2-1.0)**2+.5 * self.a*(q+1)
