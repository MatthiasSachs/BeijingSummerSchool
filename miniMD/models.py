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



class HarmonicOscillator(Model):       
        
    def __init__(self, dim=1, k=1.0, rp=0.0):
        
        super(HarmonicOscillator,self).__init__(dim)
        
        self.rp_vec = rp * np.ones(self.dim)
        self.k_vec = k * np.ones(self.dim)
    
        

    def comp_force(self, x):
        return - self.k_vec * (x - self.rp_vec)
        
    def comp_potential(self, x):
        return .5 * np.sum( self.k_vec * (x - self.rp_vec)**2)
        
