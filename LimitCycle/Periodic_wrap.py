import numpy as np

import Periodic as p

from base import base

class Periodic_wrap(base):
    """ Class to wrap many of the functions in the Periodic c++ class.
    Hopefully can eventually be replaced with python versions to reduce
    the dependence on older code."""

    def __init__(self, model, paramset, y0):
        base.__init__(model, paramset, y0)

        self.gen_options = {
            'bvp_ftol'

    def pClassSetup(self):
        """
        Sets up a new Periodic subclass. This class is a c++ class I
        wrote and wrapped with python for specific calculations
        (rootfinding, bvp solution) that were not available in casadi.
        """
        if hasattr(self,'pClass'): del self.pClass
        self.pClass = p.Periodic(self.modlT)
        self.sety0(self.y0)
        self.setparamset(self.paramset)
        self.pClass.setFTOL(self.intoptions['bvp_ftol'])
        self.pClass.setFINDY0TOL(self.intoptions['y0tol'])
    
    def sety0(self,y0):
        """
        Iterface with c++ class, sets y0 initial condition.
        """
        for j in range(0,self.NEQ+1): self.pClass.setIC(y0[j],j)

    def gety0(self):
        """
        Utility function to pull y0 from Periodic sublcass (self.pClass)
        and return a numpy array
        """
        return np.array([self.pClass.getIC(i) for i in
                         range(0,self.NEQ+1)])
        
    def setparamset(self,paramset):
        """
        Iterface with c++ class, sets parameter set.
        """
        for j in range(0,self.NP):
            self.pClass.setParamset(paramset[j],j)
            
    def findy0(self, tout=300, tol=1E-3):
        """
        Call periodic.findy0 to integrate the solution until ydot[0] =
        0. Should be maximum value
 
        Parameters
        ----------
        tout : float or int
            Maximum integration time. Should be approximately 4-5
            oscillatory periods
        """

        self.pClassSetup()
        self.pClass.setFINDY0TOL(tol)
        out = self.pClass.findy0(tout)
        if out > 0:
           if out is 1: raise RuntimeError("findy0 failed: setup")
           if out is 2: raise RuntimeError("findy0 failed: CVode Error")
           if out is 3: raise RuntimeError("findy0 failed: Roots Error")
        self.y0 = self.gety0()
        
        if self.y0[-1] < 0:
            self.y0[-1] = 1
            raise RuntimeWarning("findy0: not converged")

    def solveBVP_periodic(self):
        """
        call periodic.bvp to solve the boundary value problem for the
        exact limit cycle
        """
        
        self.pClassSetup()
        if self.pClass.bvp():
            raise RuntimeError("bvpsolve: Failed to Converge")
        else: self.y0 = self.gety0()
    
    def roots(self):
        """
        call periodic.roots to obtain the full max/min values and times
        for each state in the system. 
        """
        self.pClassSetup()
        assert self.pClass.roots() == 0, "pBase: Root fn failed"
            
        self.Ymax = np.array([self.pClass.getYmax(j,j) for j in
                              range(0,self.NEQ)])
        self.Ymin = np.array([self.pClass.getYmin(j,j) for j in
                              range(0,self.NEQ)])
        self.Tmax = np.array([self.pClass.getTmax(j) for j in
                              range(0,self.NEQ)])*self.y0[-1]
        self.Tmin = np.array([self.pClass.getTmin(j) for j in
                              range(0,self.NEQ)])*self.y0[-1]
        
        flag = True
        if any(self.Ymax == -1) or any(self.Ymin == -1):
            inds = np.where(self.Ymax == -1)[0]
            self.Ymax[inds] = self.y0[inds]
            self.Ymin[inds] = self.y0[inds]
            flag = False

        return flag
