# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:23:39 2016

@author:    Matthias Sachs (email: msachs@math.duke.edu) 
            Anton Martinsson (email: Anton.Martinsson@ed.ac.uk)

Copyright: Duke University & Univeristy of Edinburgh

Please contact the authors if you would like to reuse the code outside 
of the tutorial session
"""
import numpy as np
import abc


       
class Integrator(object):
    __metaclass__ = abc.ABCMeta
    '''
    Base class for numerical integrators
    '''
    
    def __init__(self, model, h):
        '''
        :params model: object of class Model
        :params h: stepsize used for integration
        '''
        self.model = model 
        self.h = h
        self.q = np.zeros(self.model.dim)
        self.force= np.zeros(self.model.dim) 
        self.outputsheduler = None

    
    def initialise(self, initial_values=None):
        if initial_values is not None:
            for key in initial_values.keys():
                setattr(self, key, initial_values[key])
        self.update_dyn_values();     
        
    def update_dyn_values(self):          
        self.force = self.model.comp_force(self.q)
        
    def run(self, initial_values=None):
        self.initialise(initial_values)
        self.outputsheduler.feed(0)
        for t in range(self.outputsheduler.Nsteps):
            self.traverse()
            self.outputsheduler.feed(t+1)
        
    def traverse(self):
        '''
        Stepper function. Specific to each integrator
        '''
        raise NotImplementedError()
         
    def print_summary(self):
        pass
    

class HamDynIntegrator(Integrator):
    __metaclass__ = abc.ABCMeta
    '''
    Base class for numerical integrators for Hamiltonian systems
    '''
    
    def __init__(self, model, h):
        super(HamDynIntegrator,self).__init__(model, h)
        self.p = np.zeros(self.model.dim)
        
class EulerHamDyn(HamDynIntegrator):
    '''
    Standard Euler integration scheme for Hamiltonian system
    ''' 
    def traverse(self):
        
        # Update gradient force      
        self.force = self.model.comp_force(self.q)
        # Update q
        self.q += self.h * self.p
        # Update p
        self.p += self.h * self.force


class VelocityVerlet(HamDynIntegrator):
    '''
    Velocity Verlet integration scheme for Hamiltonian system
    ''' 
    def traverse(self):
    
        # Update p
        self.p += .5 * self.h * self.force
        # Update q
        self.q += self.h * self.p
        # Update gradient force      
        self.force = self.model.comp_force(self.q)
        # Update p
        self.p += .5 * self.h * self.force
        
'''
Place implementations for SymplecticEuler and VelocityVerlet here

'''        

class RESPA(HamDynIntegrator):
    
    def __init__(self, model, h, r=1):
        '''
        :params model: object of class MultiScaleModel
        :params h: stepsize used for integration
        '''
        super(RESPA,self).__init__(model, h)
        self.r = r
        self.h_substep = float(self.h) /self.r
        self.fast_force = np.zeros(self.force.shape)
        self.slow_force = np.zeros(self.force.shape)
        
    def update_dyn_values(self):          
        self.force = self.model.comp_force(self.q)
        self.fast_force = self.model.comp_fastForce(self.q)
        self.slow_force = self.model.comp_slowForce(self.q)
        
    def traverse(self):
        
        # Update slow component of p
        self.p += .5 * self.h * self.slow_force
        # RESPA loop
        for i in range(self.r):
            # Update fast component of p
            self.p += .5 * self.h * self.fast_force
            # Update q
            self.q += self.h_substep * self.p
            # Update fast force
            self.fast_force = self.model.comp_fastForce(self.q)
            # Update fast component of p
            self.p += .5 * self.h * self.fast_force
        # Update slow force      
        self.slow_force = self.model.comp_slowForce(self.q)
        # Update slow component of p
        self.p += .5 * self.h * self.slow_force



class Thermostat(Integrator):
    __metaclass__ = abc.ABCMeta
    '''
    Base class for first order (position only) stochastic dynamics
    '''
    def __init__(self, model, h, Tk_B=1.0):
        '''
        :param h: discrete stepsize for time
        :param model: class which is used to update the force
        :param Tk_B: temperature paramameter
        '''
        super(Thermostat,self).__init__(model, h)
        self.Tk_B = None
        self.set_Tk_B(Tk_B)
        
    def set_Tk_B(self, Tk_B):
        self.Tk_B = Tk_B

class BrownianDynamics(Thermostat):
    __metaclass__ = abc.ABCMeta
    '''
    Base class for numerical integrators for Brownian dynamics 
    '''
    def __init__(self, model, h, Tk_B=1.0):
        super(BrownianDynamics,self).__init__(model, h, Tk_B)
    


class EulerMaruyamaBD(BrownianDynamics):
    """ 
        Class which implements the Euler-Maruyama method
        for Brownian Dynamics / Overdamped Langevin dynamics
    """
    
    def __init__(self,model, h, Tk_B):
        """ Init function for the class
        
        :param h: discrete stepsize for the time discretization
        :param model: class which is used to update the force
        :param Tk_B: temperature paramameter
        """
        
        super(EulerMaruyamaBD, self).__init__(model, h, Tk_B)

        self.zeta = np.sqrt(2. * self.h * self.Tk_B)
     
    def traverse(self):
        # step method forward in time
        self.q += self.h * self.force + self.zeta * np.random.normal(0., 1., self.model.dim)
        # force update
        self.model.apply_boundary_conditions(self.q)
        self.force = self.model.comp_force(self.q)

class HeunsMethodBD(BrownianDynamics):
    """ 
    Class which implements Heun's method
    for Brownian Dynamics / Overdamped Langevin dynamics
    """
    def __init__(self, model, h, Tk_B):
        """ Init function for the class
        
        :param h: discrete stepsize for the time discretization
        :param model: class which is used to update the force
        :param Tk_B: temperature paramameter
        """        
        super(HeunsMethodBD, self).__init__(model, h, Tk_B)

        self.zeta = np.sqrt(2. * self.h * self.Tk_B)
    
    def traverse(self):
        """ Integration function which evolves the model
        given in self._model
        """
        # cache some data 
        noise_cache = np.random.normal(0., 1., self.model.dim)

        # preforce update
        q_cache = self.q + self.h * self.force + self.zeta * noise_cache

        # force update #1
        force_cache = self.model.comp_force(q_cache)
        
        # post intermediate force update
        self.q += .5 * self.h * (force_cache + self.force) + self.zeta * noise_cache
        
        # force update #2
        self.model.apply_boundary_conditions(self.q)
        self.force = self.model.comp_force(self.q)

class LeimkuhlerMatthews(BrownianDynamics):
    """ 
    Class which the Leimkuhler-Matthews method
    for Brownian Dynamics / Overdamped Langevin dynamics
    """
        
        
class KineticThermostat(Thermostat):
    __metaclass__ = abc.ABCMeta
    """
    Base class for kinetic thermostats (thermostat methods possesing a momentum
    variable)
    """
    def __init__(self, model, h, Tk_B=1.0):
        '''
        :param Tk_B: temperature paramameter
        
        for other parameters see parent class
        '''
        super(KineticThermostat,self).__init__(model, h)
        self.p = np.zeros(self.model.dim)
        
class LangevinThermostat(KineticThermostat):
    __metaclass__ = abc.ABCMeta
    """
    Base class for thermostats implementing the underdamped Langevin equation 
    """
    
    def __init__(self, model, h, Tk_B=1.0, gamma=1.0):
        """ Init function for the class
        '''
        :param gamma:   friction coefficient
                        
        for other parameters see parent class
        '''
        """
        super(KineticThermostat,self).__init__(model, h, Tk_B)
        self.gamma = gamma
        
        
