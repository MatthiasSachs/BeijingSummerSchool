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
                setattr(self, key, np.copy(initial_values[key]))
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
    def __init__(self, model, h, Tk_B):
        super(LeimkuhlerMatthews, self).__init__(model, h, Tk_B)
        self.noise_k1 = np.random.normal(0., 1., self.model.dim)
        self.zeta = np.sqrt(.5 * self.h * self.Tk_B )
    
    def traverse(self):
        """ Integration function which evolves the model
            given in self._model
            """
        noise_kp1 = np.random.normal(0., 1., self.model.dim)
        
        # pre force update
        self.q += self.h * self.force + self.zeta * (self.noise_k1 + noise_kp1)
        
        # force update
        self.model.apply_boundary_conditions(self.q)
        self.force = self.model.comp_force(self.q)
        
        # update the chached noise
        self.noise_k1 = noise_kp1
        
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
        super(LangevinThermostat,self).__init__(model, h, Tk_B)
        self.gamma = gamma
        
class LangevinBAOSplitting(LangevinThermostat):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, model, h, Tk_B=1.0, gamma=1.0):
        super(LangevinBAOSplitting,self).__init__(model, h, Tk_B, gamma)
        
        self.alpha = np.exp(-self.h * self.gamma)
        self.zeta = np.sqrt((1.0-self.alpha**2)*self.Tk_B)
        self.alpha2 = np.exp(-.5 * self.h * self.gamma)
        self.zeta2 = np.sqrt((1.0-self.alpha2**2)*self.Tk_B)
        
class LangevinBAOAB(LangevinBAOSplitting):
    
    def traverse(self):
        self.p += .5 * self.h * self.force
        self.q += .5 * self.h * self.p
        self.p = self.alpha * self.p + self.zeta * np.random.normal(0,1.0, self.model.dim)
        self.q += .5 * self.h * self.p
        self.force = self.model.comp_force(self.q)
        self.p += .5 * self.h * self.force
        
        
class LangevinOBABO(LangevinBAOSplitting):
    
    def traverse(self):
        self.p = self.alpha2 * self.p + self.zeta2 * np.random.normal(0,1.0, self.model.dim)
        self.p += .5 * self.h * self.force
        self.q += self.h * self.p
        self.force = self.model.comp_force(self.q)
        self.p += .5 * self.h * self.force
        self.p = self.alpha2 * self.p + self.zeta2 * np.random.normal(0,1.0, self.model.dim)

        
class EnsembleQuasiNewton(LangevinBAOSplitting):
    """ Sampler implementing ensemble Quasi Newton method 
    implementation is witout local weighting of estimates of the walker covariance
    """
    
    def __init__(self, repmodel, h, Tk_B=1.0, gamma=1.0, regparams=1.0, B_update_mod=1):

        
        super(EnsembleQuasiNewton, self).__init__(repmodel, h, Tk_B=1.0, gamma=1.0)
        self.pmdim = self.model.protmodel.dim
        self.Bmatrix = np.zeros([self.model.nreplicas, self.pmdim, self.pmdim ])
        for i in range(self.model.nreplicas):
            self.Bmatrix[i,:,:] = np.eye( self.pmdim)
        self.regparams = regparams
        self.B_update_mod = B_update_mod
        self.substep_counter = 0
        
    

    def traverse(self):
        
        nreplicas = self.model.nreplicas
           
         # update preconditioner
        if self.substep_counter % self.B_update_mod == 0:
            self.update_Bmatrix()
            
        # B-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h  * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim])
            
        # A-step
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])
        
        # O-step
        self.p = self.alpha * self.p + self.zeta * np.random.normal(0., 1., self.model.dim)
         
        # A-step   
        for i in range(nreplicas):
            self.p[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self.force[i*self.pmdim:(i+1)*self.pmdim])
        
        # update force
        self.model.apply_boundary_conditions()
        self.force = self.model.comp_force(self.q)
        
        # B-step
        for i in range(nreplicas):
            self.q[i*self.pmdim:(i+1)*self.pmdim] += .5 * self.h * np.matmul(self.Bmatrix[i,:,:], self.p[i*self.pmdim:(i+1)*self.pmdim]) 
         
       
            
        self.substep_counter+=1
        
    def update_Bmatrix(self):
        if self.model.nreplicas > 1:
            indices = [i for i in range(self.model.nreplicas)]
            for r in range(self.model.nreplicas):
                mask =  np.array(indices[:r] + indices[(r + 1):])
                #print(np.cov(self.q.reshape([self.model.nreplicas,self.pmdim])[mask,:],rowvar=False) + self.regparams * np.eye(self.pmdim))
                self.Bmatrix[r,:,:] = np.linalg.cholesky(
                        np.cov(self.q.reshape([self.model.nreplicas,self.pmdim])[mask,:],rowvar=False) + self.regparams * np.eye(self.pmdim)
                                                        )

def autocorr(x, maxlag=100):
    acf_vec = np.zeros(maxlag)
    xmean = np.mean(x)
    n = x.shape[0]
    for lag in range(maxlag):
        index = np.arange(0,n-lag,1)
        index_shifted = np.arange(lag,n,1)
        acf_vec[lag] = np.mean((x[index ]-xmean)*(x[index_shifted]-xmean))
    
    return acf_vec   