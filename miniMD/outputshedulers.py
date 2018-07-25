# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 14:37:14 2016

@author:    Matthias Sachs (email: msachs@math.duke.edu) 
            Anton Martinsson (email: Anton.Martinsson@ed.ac.uk)

Copyright: Duke University & Univeristy of Edinburgh

Please contact the authors if you would like to reuse the code outside 
of the tutorial session
"""

import numpy as np
import abc

    
        
class Outputsheduler(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def feed(self, integrator, t):
        raise NotImplementedError()
        
        
        
    
class BufferedOutputsheduler(Outputsheduler):
    
        
    def __init__(self, integrator, Nsteps, varname_list=None, modprnt=1):
        ''''
        :params integrator: object of class Integrator to which the outputsheduler is attached
        
        :params Nsteps: number of time steps the integrator is run when called by 
                        the funciton 'integrator.sample()'
                        
        :params varname_list = [<string:varname1>, <string:varname2>,...] 
                    List of strings containing the names variables of 
                    in string format of the integrator object which are to be 
                    monitored when the function 'integrator.sample()' is called. 
                    The trajectories are then saved in a numpy arrays 
                    as arguments of the outputsheduler named 
                    traj_varname1, traj_varname2, ...
                    
        :params modprnt: specfies the frequency at which the output is written, 
                        i.e., if modprnt=1, then output is written at every 
                        time step. If modprn=10, the output is written at 
                        every 10th time step
                    
        
        '''
        self.integrator = integrator
        self.Nsteps = Nsteps
        
        
        self.modprnt = modprnt
            
        self.Nsamples = Nsteps // self.modprnt + 1
            
        self.varname_list = varname_list
        
    
        for varname in self.varname_list:
            var = getattr(integrator,varname)
            setattr(self,'traj_' + varname, np.zeros([self.Nsamples, var.size]) )
        self.tprnt = 0
        
        self.integrator.outputsheduler = self
        
    def feed(self, t):
        if(self.tprnt >= self.Nsamples):
                raise Exception('Warning: outputsheduler must be reinitialized after being used for one integration run')

        if t % self.modprnt == 0:
            for varname in self.varname_list:
                traj_var = getattr(self,'traj_' + varname)
                traj_var[self.tprnt,:]=getattr(self.integrator,varname)
            
            self.tprnt += 1

                
    def get_traj(self, varname):
        return getattr(self, 'traj_' + varname)
 


class BufferedOutputshedulerU(BufferedOutputsheduler):
    '''
    As BufferedOutputsheduler but collects by default the potential value of 
    self.integrator.model at every time step.
    '''
        
    def __init__(self, integrator, Nsteps, varname_list=None, modprnt=None):
        super(BufferedOutputshedulerU,self).__init__(integrator, Nsteps, varname_list, modprnt)
        self.traj_potE = np.zeros(self.Nsamples)
        self.traj_kinE = np.zeros(self.Nsamples)
        self.traj_totalE = np.zeros(self.Nsamples)
        
    def feed(self, t):
        if(self.tprnt >= self.Nsamples):
                raise Exception('Warning: outputsheduler must be reinitialized after being used for one integration run')

        if t % self.modprnt == 0:
            for varname in self.varname_list:
                traj_var = getattr(self,'traj_' + varname)
                traj_var[self.tprnt,:]=getattr(self.integrator,varname)
            self.traj_potE[self.tprnt] = self.integrator.model.comp_potential(self.integrator.q)
            self.traj_kinE[self.tprnt] = .5*np.sum(self.integrator.p**2)
            self.traj_totalE[self.tprnt] = self.traj_potE[self.tprnt] + self.traj_kinE[self.tprnt]
            self.tprnt += 1
            
class HistogramOutputsheduler(BufferedOutputsheduler):
    
    def __init__(self, integrator, Nsteps, varname_list=None, bins_list=None, modprnt=None):
        
        self.integrator = integrator
        self.Nsteps = Nsteps
        
        
        self.modprnt = modprnt
            
        self.Nsamples = Nsteps // self.modprnt + 1
            
        self.varname_list = varname_list
        self.bins_list = bins_list
    
        for i in range(len(self.varname_list)):
            varname = self.varname_list[i]
            setattr(self,'traj_bins_' + varname, np.zeros([self.Nsamples, bins_list[i].shape[0]-1]) )
        self.tprnt = 0
        
        self.integrator.outputsheduler = self
        
    def feed(self, t):
        if(self.tprnt >= self.Nsamples):
                raise Exception('Warning: outputsheduler must be reinitialized after being used for one integration run')

        if t % self.modprnt == 0:
            for i in range(len(self.varname_list)):
                varname = self.varname_list[i]
                bins = self.bins_list[i]
                traj_var = getattr(self,'traj_bins_' + varname)
                bincounts, bins = np.histogram(getattr(self.integrator, varname), bins=bins)
                traj_var[self.tprnt,:]= bincounts
            
            self.tprnt += 1