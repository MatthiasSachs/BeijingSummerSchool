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

class MVHarmonicOscillator(Model):       
        
    def __init__(self, k_vec=np.array([1.0])):
        
        super(MVHarmonicOscillator,self).__init__(dim=k_vec.shape[0])
        
        self.k_vec = k_vec
    
        

    def comp_force(self, x):
        return - self.k_vec * x 
        
    def comp_potential(self, x):
        return .5 * self.k_vec * np.sum( x**2,axis=0)
    
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

class MultiVariateGaussian(Model):
    
    def __init__(self, Omega):
        
        super(MultiVariateGaussian, self).__init__(dim=Omega.shape[0])
        self.Omega = Omega
        
        
        
        
    def comp_potential(self,q):
        """ returns the potential at the current position
        """
        return .5 * np.dot(q,np.dot(self.Omega, q))

    def comp_force(self, q):
        """ returns the force internally from the current position
        """
        return -np.dot(self.Omega, q)
    
    
    def apply_boundary_conditions(self):
        pass
    
class ReplicatedModel(Model):
    """ Baseclass for models used in samplers using multiple replicas.
    """    
    
    def __init__(self, model, nreplicas=1):
        self.protmodel = model
        self.pmdim =  self.protmodel.dim
        self.dim = self.protmodel.dim * nreplicas
        self.nreplicas = nreplicas
        #self.q = np.repeat(np.reshape(model.q, (-1, model.dim)),self.nreplicas, axis=0)
        #self.force = np.repeat(np.reshape(model.f, (-1, model.dim)),self.nreplicas, axis=0)
        #if model.p is not None:
        #    self.p = np.repeat(np.reshape(model.p, (-1, model.dim)),self.nreplicas, axis=0)
        
    def comp_potential(self,q):
        """ returns the potential at the current position
        """
        pot = 0.0
        for i in range(self.nreplicas):
            pot+= self.protmodel.comp_potential(q[i*self.pmdim:((i+1)*self.pmdim)])
            
        return pot
        
    def comp_force(self,q):
        """ updates the force internally from the current position
        """
        force = np.zeros( self.nreplicas * self.pmdim )
        for i in range(self.nreplicas):
            force[i*self.pmdim:((i+1)*self.pmdim)] = self.protmodel.comp_force(q[i*self.pmdim:((i+1)*self.pmdim)])
            
        return force
    
    
    def apply_boundary_conditions(self):
        pass # In this implementation the class ReplicatedModel is assumed to have no speficied boundary conditions


class BayesianLogisticRegression(Model):
    
    def __init__(self, prior_mean, prior_cov, data):
        
        
        super(BayesianLogisticRegression, self).__init__(dim=prior_mean.shape[0])
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.data = data
        
        self.prior_cov_inv = np.linalg.inv(self.prior_cov)
        self.prior_cov_det = np.linalg.det(self.prior_cov)
        
        
    #def as2Dvector(self, position):
     #   return np.reshape(position, (-1, self.dim))
    
    def comp_force(self, q):
        return self.grad_log_like(q) + self.grad_log_prior(q)
        
    def comp_potential(self, q):
        return self.log_like(q) + self.log_prior(q)
        
    
    def predict(self, params, x):
        return self._logistic(np.dot(x, params))
        
    def log_like(self, params):
        x, y = self.data
        return np.sum(self.predict(params, x))
    
    def grad_log_like(self, params):
        x, y = self.data
        prob0 = self.predict(params, x)
        return np.dot( (y - prob0), x)
    
    def log_prior(self, x):
        return -np.dot(x - self.prior_mean, np.dot(self.prior_cov_inv, x - self.prior_mean))/2.0
        
    def grad_log_prior(self, x):
        return -np.dot(self.prior_cov_inv, x - self.prior_mean)
    
    def _logistic(self, x):
        expx = np.exp(x)
        prob0 = np.zeros(x.shape)
        index = np.isinf(expx)
        prob0[np.logical_not(index)] = expx[np.logical_not(index)]/(1.0+expx[np.logical_not(index)])
        prob0[index] = 1.0
        
        return prob0
    
    def predict_from_sample(self, params_traj, x): # params_traj  is a T \times self.dim  matrix, where each column represents one sample of params 
        '''
        returns the probability vector 1/T sum_{t=0}^{T-1}[P(y_i| params[:,t], x)]_{ i = 0, ..., n_classes-1}  
        '''
        T = params_traj.shape[0]
        prob = np.zeros(2)
        for t in range(T):
            params = params_traj[t,:]            
            prob += self.predict(params, x)
        prob /= T
        
        return prob   
    
    def plot_prediction(self, q_trajectory, grid, Neval=100, show_training_data=True ):
        import time
        import matplotlib.pyplot as plt
        
        xx1, xx2 = grid
        t = time.time()
        z= np.zeros([len(xx1),len(xx2)])
        modthin = q_trajectory.shape[0]//Neval
        for i in range(len(xx1)):
            for j in  range(len(xx2)):
                x = np.array([xx1[i],xx2[j]])
                z[i,j] = self.predict_from_sample(q_trajectory[::modthin,:self.dim], x)[0]

        elapsed = time.time() - t
        
        print('Time to calculate predictions: {}'.format(elapsed))

        fig2, ax2 = plt.subplots()
        ax2.pcolor(xx1, xx2, z.transpose(), cmap='RdBu', vmin=0, vmax=1)
        cax = ax2.pcolor(xx1, xx2, z.transpose(), cmap='RdBu', vmin=0, vmax=1)
        cbar = fig2.colorbar(cax)
        cbar.ax.set_ylabel('$\mathbb{P}(Y = 1)$')
    
        if show_training_data:
            '''
            Include training data
            '''
            X,Y = self.data
            ndata = Y.shape[0]
            color_dict= {0:'red', 1 :'blue'}
            colors = [color_dict[Y[i]] for i in range(ndata)]
            
            ax2.scatter(X[:,0],X[:,1], c=colors)
        
        ax2.set_title('Prediction')
        ax2.set_xlabel('$y-coordinate$')
        ax2.set_ylabel('$x-coordinate$')
        return fig2, ax2    