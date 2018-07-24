#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:36:17 2018

@author: msachs2
"""
from miniMD import models
from miniMD import integrators
from miniMD import outputshedulers as outp

import numpy as np
import matplotlib.pyplot as plt
import abc
    
    
class Constraint(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, ncons):
        self.ncons = ncons    #number of constraints

        
    def Dgfunc(self, q):
        raise NotImplementedError()
    
    def gfunc(self, q):
        raise NotImplementedError()

class DPConstraint(Constraint):
    
    def __init__(self, r1=1.0):
        super(DPConstraint, self).__init__(ncons = 1)
        self.r1 = r1
        
    def gfunc(self, q):
        return (np.sum(q[:2]**2) - self.r1**2).reshape([1,1])
        
    def Dgfunc(self,q):
        G = np.zeros([1,4])
        G[0,:2] = 2.0 *q[:2]
        return G
    
class CIntegrator(integrators.HamDynIntegrator): 
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model, h, constraint):
        super(CIntegrator, self).__init__(model, h)
        self.constraint = constraint
             
        
class Rattle(CIntegrator): 
     
    def __init__(self, model, h, constraint, tol=0.0001, maxit=float('inf')):
        super(Rattle, self).__init__(model, h,  constraint)
        
        self.p = np.zeros(self.q.shape)
        self.tol = tol
        self.maxit = maxit
        self.g = None
        self.G = None
        self.S = None
        self.SLchol = None
                         
    
    def update_dyn_values(self):  
        super(Rattle, self).update_dyn_values()
        self.update_constraints()
    
    def update_constraints(self):
        self.G = self.constraint.Dgfunc(self.q)
        self.S = np.dot(self.G,np.transpose(self.G))
        self.SLchol = np.linalg.cholesky(self.S)
        
    def shake_newton_iteration(self):
        
        error = float('inf')
        i = 0
        if self.constraint.ncons > 0:
            while (i < self.maxit and self.tol <= error ):
                self.g = self.constraint.gfunc(self.q)
                dlambda = np.linalg.solve(np.transpose(self.SLchol),np.linalg.solve(self.SLchol, self.g))
                delta = np.dot(np.transpose(self.G),dlambda)
                self.q += - delta.flatten()
                #self.p += - delta/self.h
                self.p += - delta.flatten()/self.h
                error = np.linalg.norm(dlambda)
                i += 1
        if (i == self.maxit):
            print('Rattle: No convergence in SHAKE iteration');
  
    def rattle_iteration(self):
        '''
        Rattle
        '''
        if self.constraint.ncons > 0:
            b = np.dot(self.G, self.p)
            x = np.linalg.solve(np.transpose(self.SLchol),np.linalg.solve(self.SLchol, b))
            self.p += - np.dot(np.transpose(self.G),x)
    
    def traverse(self):
        
        '''
        half B-step
        '''
        self.p += .5 * self.h * self.force
        '''
        full A-step
        '''
        self.q +=  self.h * self.p
        #self.model.apply_boundary_condition(self.q)
        '''
        Shake-Newton iteration
        '''
        self.shake_newton_iteration()  
        
        '''
        update forces and constraints
        '''
        self.update_dyn_values()
        #self.update_constraints()
        
        '''
        B-step
        '''   
        self.p += .5 * self.h * self.force
        
        '''
        Rattle
        '''
        self.rattle_iteration()
