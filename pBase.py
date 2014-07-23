from __future__ import division
from time import time

import numpy as np
import casadi as cs

import Periodic as p

from Utilities import MultivariatePeriodicSpline

# TODO: update all of the tolerances to a dictionary. In fact, this
# whole class should probably get updated.


class pBase(object):
    """
    Base class for most periodic ode derived classes. Mixes together my
    swig-wrapped c++ "Periodic" class for specific periodic calculations
    and the casadi suite for general integrations.
    """
        
    def __init__(self, model, paramset, y0=None):
        """
        Sets up pBase with the required information

        parameters
        ----------
        model : casadi.sxfunction
            model equations, sepecified through an integrator-ready
            casadi sx function
        paramset : iterable
            parameters for the model provided. Must be the correct length.
        y0 : (optional) iterable
            Initial conditions, specifying where d(y[0])/dt = 0
            (maximum) for the first state variable.
        """
       
        self.model = model
        self.NEQ = self.model.input(cs.DAE_X).size()
        self.NP  = self.model.input(cs.DAE_P).size()
        self.modifiedModel()
        
        self.model.init()
        self.jacp = self.model.jacobian(cs.DAE_P,0); self.jacp.init()
        self.jacy = self.model.jacobian(cs.DAE_X,0); self.jacy.init()
        
        self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.NEQ)]
        self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.NP)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.NP)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.NEQ)):
            self.ydict[par] = ind
        
        self.paramset = paramset

        self.intoptions = {
            'y0tol'            : 1E-3,
            'bvp_ftol'         : 1E-10,
            'bvp_abstol'       : 1E-12,
            'bvp_reltol'       : 1E-10,
            'sensabstol'       : 1E-11,
            'sensreltol'       : 1E-9,
            'sensmaxnumsteps'  : 80000,
            'sensmethod'       : 'staggered',
            'transabstol'      : 1E-4,
            'transreltol'      : 1E-4,
            'transmaxnumsteps' : 5000,
            'lc_abstol'        : 1E-11,
            'lc_reltol'        : 1E-9,
            'lc_maxnumsteps'   : 40000,
            'lc_res'           : 200,
            'int_abstol'       : 1E-8,
            'int_reltol'       : 1E-6,
        }
            
        # No initial conditions provided, bvp solution initiated
        if y0 is None:
            self.y0 = 5*np.ones(self.NEQ+1)
            self.calcY0(600)
        else: self.y0 = y0

    # Shortcut methods
    def _phi_to_t(self, phi): return phi*self.y0[-1]/(2*np.pi)
    def _t_to_phi(self, t): return (2*np.pi)*t/self.y0[-1]

    def lc_phi(self, phi):
        """ interpolate the selc.lc interpolation object using a time on
        (0,2*pi) """
        return self.lc(self._phi_to_t(phi%(2*np.pi)))

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

    def modifiedModel(self):
        """
        Creates a new casadi model with period as a parameter, such that
        the model has an oscillatory period of 1. Necessary for the
        exact determinination of the period and initial conditions
        through the BVP method. (see Wilkins et. al. 2009 SIAM J of Sci
        Comp)
        """

        pSX = self.model.inputSX(cs.DAE_P)
        T = cs.ssym("T")
        pTSX = cs.vertcat([pSX, T])
        
        self.modlT = cs.SXFunction(
            cs.daeIn(t=self.model.inputSX(cs.DAE_T),
                     x=self.model.inputSX(cs.DAE_X), p=pTSX),
            cs.daeOut(ode=cs.vertcat([self.model.outputSX()*T])))

        self.modlT.setOption("name","T-shifted model")

    def findstationary(self, guess=None):
        """
        Find the stationary points dy/dt = 0, and check if it is a
        stable attractor (non oscillatory).

        Parameters
        ----------
        guess : (optional) iterable
            starting value for the iterative solver. If empty, uses
            current value for initial condition, y0.

        Returns
        -------
        +0 : Fixed point is not a steady-state attractor
        +1 : Fixed point IS a steady-state attractor
        -1 : Solution failed to converge
        """
        try:
            self.corestationary(guess)
            if all(np.real(self.eigs) < 0): return 1
            else: return 0

        except Exception: return -1

    def corestationary(self,guess=None):
        """
        find stationary solutions that satisfy ydot = 0 for stability
        analysis. 
        """
        if guess is None: guess = np.array(self.y0[:-1])
        else: guess = np.array(guess)
        y = self.model.inputSX(cs.DAE_X)
        t = self.model.inputSX(cs.DAE_T)
        p = self.model.inputSX(cs.DAE_P)
        ode = self.model.outputSX()
        fn = cs.SXFunction([y,t,p],[ode])
        kfn = cs.KinsolSolver(fn)
        abstol = 1E-10
        kfn.setOption("abstol",abstol)
        kfn.setOption("constraints",(2,)*self.NEQ)
        kfn.setOption("linear_solver","dense")
        kfn.setOption("numeric_jacobian",True)
        kfn.setOption("u_scale",(100/guess).tolist())
        kfn.setOption("numeric_hessian",True)
        kfn.setOption("disable_internal_warnings",True)
        kfn.init()
        kfn.setInput(self.paramset,1)
        kfn.setOutput(guess)
        kfn.evaluate()
        y0out = kfn.output().toArray()
        
        if any(np.isnan(y0out)):
            raise RuntimeError("findstationary: KINSOL failed to find \
                               acceptable solution")
        
        self.ss = y0out.flatten()
        
        if np.linalg.norm(self.dydt(self.ss)) >= abstol or any(y0out <= 0):
            raise RuntimeError("findstationary: KINSOL failed to reach \
                               acceptable bounds")
              
        self.eigs = np.linalg.eigvals(self.dfdy(self.ss))
        

    def intPastTrans(self,tf=500):
        """
        integrate the solution until self.tf
        """
        
        self.integrator = cs.CVodesIntegrator(self.model)
        self.integrator.setOption("abstol", self.intoptions['transabstol'])
        self.integrator.setOption("reltol", self.intoptions['transreltol'])
        self.integrator.setOption("tf", tf)
        self.integrator.setOption("max_num_steps",
                                  self.intoptions['transmaxnumsteps'])
        self.integrator.setOption("disable_internal_warnings", True)
        self.integrator.init()
        self.integrator.setInput((self.y0[:-1]), cs.INTEGRATOR_X0)
        self.integrator.setInput(self.paramset, cs.INTEGRATOR_P)
        self.integrator.evaluate()
        self.y0 = \
        np.array(self.integrator.output().toArray().squeeze().tolist() +
                 [1])
        
        if np.abs(self.model.output().toArray()).sum() <= 1E-3*self.NEQ:
            raise RuntimeWarning("intPastTrans: converged to steady state")
            
    def approxY0(self, tout=300, tol=1E-3):
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

    def solveBVP(self, method='scipy', backup='periodic'):
        """
        Chooses between available solver methods to solve the boundary
        value problem. Backup solver invoked in case of failure
        """
        available = {
            'periodic' : self.solveBVP_periodic,
            'casadi'   : self.solveBVP_casadi,
            'scipy'    : self.solveBVP_scipy}

        y0in = np.array(self.y0)

        try: return available[method]()
        except Exception:
            self.y0 = np.array(y0in)
            try: return available[backup]()
            except Exception:
                self.y0 = y0in
                self.approxY0(tol=1E-4)
                return available[method]()


    def solveBVP_periodic(self):
        """
        call periodic.bvp to solve the boundary value problem for the
        exact limit cycle
        """
        
        self.pClassSetup()
        if self.pClass.bvp():
            raise RuntimeError("bvpsolve: Failed to Converge")
        else: self.y0 = self.gety0()
    
    def solveBVP_casadi(self):
        """
        Uses casadi's interface to sundials to solve the boundary value
        problem using a single-shooting method with automatic differen-
        tiation. 
        """

        # Here we create and initialize the integrator SXFunction
        self.bvpint = cs.CVodesIntegrator(self.modlT)
        self.bvpint.setOption('abstol',self.intoptions['bvp_abstol'])
        self.bvpint.setOption('reltol',self.intoptions['bvp_reltol'])
        self.bvpint.setOption('tf',1)
        self.bvpint.setOption('disable_internal_warnings', True)
        self.bvpint.setOption('fsens_err_con', True)
        self.bvpint.init()

        # Vector of unknowns [y0, T]
        V = cs.msym("V",self.NEQ+1)
        y0 = V[:-1]
        T = V[-1]
        t = cs.msym('t')
        param = cs.vertcat([self.paramset, T])

        yf = self.bvpint.call(cs.integratorIn(x0=y0,p=param))[0]
        fout = self.modlT.call(cs.daeIn(t=t, x=y0,p=param))[0]

        obj = (yf - y0)**2
        obj.append(fout[0])

        F = cs.MXFunction([V],[obj])
        F.init()

        solver = cs.KinsolSolver(F)
        solver.setOption('abstol',self.intoptions['bvp_ftol'])
        solver.setOption('ad_mode', "forward")
        solver.setOption('strategy','linesearch')
        solver.setOption('numeric_jacobian', True)
        solver.setOption('exact_jacobian', False)
        solver.setOption('pretype', 'both')
        solver.setOption('use_preconditioner', True)
        solver.setOption('numeric_hessian', True)
        solver.setOption('constraints', (2,)*(self.NEQ+1))
        solver.setOption('verbose', False)
        solver.setOption('sparse', False)
        solver.setOption('linear_solver', 'dense')
        solver.init()
        solver.output().set(self.y0)
        solver.solve()

        self.y0 = solver.output().toArray().squeeze()

    def solveBVP_scipy(self, root_method='hybr'):
        """
        Use a scipy optimize function to optimize the BVP function
        """

        # Make sure inputs are the correct format
        paramset = list(self.paramset)

        
        # Here we create and initialize the integrator SXFunction
        self.bvpint = cs.CVodesIntegrator(self.modlT)
        self.bvpint.setOption('abstol',self.intoptions['bvp_abstol'])
        self.bvpint.setOption('reltol',self.intoptions['bvp_reltol'])
        self.bvpint.setOption('tf',1)
        self.bvpint.setOption('disable_internal_warnings', True)
        self.bvpint.setOption('fsens_err_con', True)
        self.bvpint.init()

        def bvp_minimize_function(x):
            """ Minimization objective """
            # perhaps penalize in try/catch?
            if np.any(x < 0): return np.ones(3)
            self.bvpint.setInput(x[:-1], cs.INTEGRATOR_X0)
            self.bvpint.setInput(paramset + [x[-1]], cs.INTEGRATOR_P)
            self.bvpint.evaluate()
            out = x[:-1] - self.bvpint.output().toArray().flatten()
            out = out.tolist()

            self.modlT.setInput(x[:-1], cs.DAE_X)
            self.modlT.setInput(paramset + [x[-1]], 2)
            self.modlT.evaluate()
            out += self.modlT.output()[0].toArray()[0].tolist()
            return np.array(out)
        
        from scipy.optimize import root

        options = {}

        root_out = root(bvp_minimize_function, self.y0,
                        tol=self.intoptions['bvp_ftol'],
                        method=root_method, options=options)

        # Check solve success
        if not root_out.status:
            raise RuntimeError("bvpsolve: " + root_out.message)

        # Check output convergence
        if np.linalg.norm(root_out.qtf) > self.intoptions['bvp_ftol']*1E4:
            raise RuntimeError("bvpsolve: nonconvergent")

        # save output to self.y0
        self.y0 = root_out.x


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

    def limitCycle(self):
        """
        integrate the solution for one period, remembering each of time
        points along the way
        """
        
        self.ts = np.linspace(0, self.y0[-1], self.intoptions['lc_res'])
        
        intlc = cs.CVodesIntegrator(self.model)
        intlc.setOption("abstol"       , self.intoptions['lc_abstol'])
        intlc.setOption("reltol"       , self.intoptions['lc_reltol'])
        intlc.setOption("max_num_steps", self.intoptions['lc_maxnumsteps'])
        intlc.setOption("tf"           , self.y0[-1])

        intsim = cs.Simulator(intlc, self.ts)
        intsim.init()
        
        # Input Arguments
        intsim.setInput(self.y0[:-1], cs.INTEGRATOR_X0)
        intsim.setInput(self.paramset, cs.INTEGRATOR_P)
        intsim.evaluate()
        self.sol = intsim.output().toArray()

        # create interpolation object
        self.lc = self.interp_sol(self.ts, self.sol)


    def interp_sol(self, tin, yin):
        """
        Function to create a periodic spline interpolater
        """
    
        return MultivariatePeriodicSpline(tin, yin, period=self.y0[-1])


    def calcY0(self,trans=300):
        """
        meta-function to call each calculation function in order for
        unknown y0. Invoked when initial condition is unknown.
        """

        try: del self.pClass
        except AttributeError: pass
        self.intPastTrans(trans)
        self.approxY0(trans)
        self.solveBVP()
        self.roots()

    def dydt(self,y):
        """
        Function to calculate model for given y.
        """
        try:
            out = []
            for yi in y:
                assert len(yi) == self.NEQ
                self.model.setInput(yi,cs.DAE_X)
                self.model.setInput(self.paramset,cs.DAE_P)
                self.model.evaluate()
                out += [self.model.output().toArray().flatten()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.model.setInput(y,cs.DAE_X)
            self.model.setInput(self.paramset,cs.DAE_P)
            self.model.evaluate()
            return self.model.output().toArray().flatten()

        
    def dfdp(self,y,p=None):
        """
        Function to calculate model jacobian for given y and p.
        """
        if p is None: p = self.paramset

        try:
            out = []
            for yi in y:
                assert len(yi) == self.NEQ
                self.jacp.setInput(yi,cs.DAE_X)
                self.jacp.setInput(p,cs.DAE_P)
                self.jacp.evaluate()
                out += [self.jacp.output().toArray()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.jacp.setInput(y,cs.DAE_X)
            self.jacp.setInput(p,cs.DAE_P)
            self.jacp.evaluate()
            return self.jacp.output().toArray()

        
    def dfdy(self,y,p=None):
        """
        Function to calculate model jacobian for given y and p.
        """
        if p is None: p = self.paramset
        try:
            out = []
            for yi in y:
                assert len(yi) == self.NEQ
                self.jacy.setInput(yi,cs.DAE_X)
                self.jacy.setInput(p,cs.DAE_P)
                self.jacy.evaluate()
                out += [self.jacy.output().toArray()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.jacy.setInput(y,cs.DAE_X)
            self.jacy.setInput(p,cs.DAE_P)
            self.jacy.evaluate()
            return self.jacy.output().toArray()
    
    def parambykey(self, key):
        """
        Get parameter value by string label or index.
        """

        try: return np.array(self.paramset)[key]
        except ValueError: return self.paramset[self.pdict[key]]
        
    def average(self):
        """
        integrate the solution with quadrature to find the average 
        species concentration. outputs to self.avg
        """
        
        ffcn_in = self.model.inputsSX()
        ode = self.model.outputSX()
        quad = cs.vertcat([ffcn_in[cs.DAE_X], ffcn_in[cs.DAE_X]**2])

        quadmodel = cs.SXFunction(ffcn_in, cs.daeOut(ode=ode, quad=quad))

        qint = cs.CVodesIntegrator(quadmodel)
        qint.setOption("abstol"        , self.intoptions['lc_abstol'])
        qint.setOption("reltol"        , self.intoptions['lc_reltol'])
        qint.setOption("max_num_steps" , self.intoptions['lc_maxnumsteps'])
        qint.setOption("tf",self.y0[-1])
        qint.init()
        qint.setInput(self.y0[:-1], cs.INTEGRATOR_X0)
        qint.setInput(self.paramset, cs.INTEGRATOR_P)
        qint.evaluate()
        quad_out = qint.output(cs.INTEGRATOR_QF).toArray().squeeze()
        self.avg = quad_out[:self.NEQ]/self.y0[-1]
        self.rms = np.sqrt(quad_out[self.NEQ:]/self.y0[-1])
        self.std = np.sqrt(self.rms**2 - self.avg**2)
        



    def offset_t(self, t_offset):
        """
        Change the time basis of the output solution in the event that
        maximum of first state is not t=0. Initial conditions in self.y0
        are left unchanged, while the ts, sol, and lc variables are
        updated to reflect the new time basis. ts no longer gaurenteed
        to fall on linspace(0, T).
        """

        # Ensure [0,T)
        t_offset = t_offset%self.y0[-1]

        # Add offset to TS, wrap about origin.
        self.ts = (self.ts[:-1] - t_offset)%self.y0[-1]

        rearrange = self.ts.argsort()
        self.ts = self.ts[rearrange]
        self.sol = self.sol[:-1][rearrange]

        # recreate interpolation object
        self.lc = self.interp_sol(self.ts, self.sol)

        # Update root info
        if hasattr(self, 'Tmax'):
            self.Tmax = (self.Tmax + t_offset)%self.y0[-1]
            self.Tmin = (self.Tmin + t_offset)%self.y0[-1]


    def check_monodromy(self):
        """
        Check the stability of the limit cycle by finding the
        eigenvalues of the monodromy matrix
        """

        integrator = cs.CVodesIntegrator(self.model)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", self.y0[-1])
        integrator.setOption("numeric_jacobian", True)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        integrator.setInput(self.y0[:-1], cs.INTEGRATOR_X0)
        integrator.setInput(self.paramset, cs.INTEGRATOR_P)

        intdyfdy0 = integrator.jacobian(cs.INTEGRATOR_X0, cs.INTEGRATOR_XF)
        intdyfdy0.evaluate()
        monodromy = intdyfdy0.output().toArray()        
        self.monodromy = monodromy

        # Calculate Floquet Multipliers, check if all (besides n_0 = 1)
        # are inside unit circle
        eigs = np.linalg.eigvals(monodromy)
        self.floquet_multipliers = np.abs(eigs)
        self.floquet_multipliers.sort()
        idx = (np.abs(self.floquet_multipliers - 1.0)).argmin()
        f = self.floquet_multipliers.tolist()
        f.pop(idx)
        
        return np.all(np.array(f) < 1)



    def first_order_sensitivity(self):
        """
        Function to calculate the first order period sensitivity
        matricies using the direct method. See Wilkins et. al. Only
        calculates initial conditions and period sensitivities.
        Functions for amplitude sensitivitys remain in sensitivityfns
        """

        self.check_monodromy()
        monodromy = self.monodromy

        integrator = cs.CVodesIntegrator(self.model)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", self.y0[-1])
        integrator.setOption("numeric_jacobian", True)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        integrator.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        integrator.setInput(self.paramset,cs.INTEGRATOR_P)

        intdyfdp = integrator.jacobian(cs.INTEGRATOR_P, cs.INTEGRATOR_XF)
        intdyfdp.evaluate()
        s0 = intdyfdp.output().toArray()

        self.model.init()
        self.model.setInput(self.y0[:-1],cs.DAE_X)
        self.model.setInput(self.paramset,cs.DAE_P)
        self.model.evaluate()
        ydot0 = self.model.output().toArray().squeeze()
        
        LHS = np.zeros([(self.NEQ + 1), (self.NEQ + 1)])
        LHS[:-1,:-1] = monodromy - np.eye(len(monodromy))
        LHS[-1,:-1] = self.dfdy(self.y0[:-1])[0]
        LHS[:-1,-1] = ydot0
        
        RHS = np.zeros([(self.NEQ + 1), self.NP])
        RHS[:-1] = -s0
        RHS[-1] = self.dfdp(self.y0[:-1])[0]
        
        unk = np.linalg.solve(LHS,RHS)
        self.S0 = unk[:-1]
        self.dTdp = unk[-1]
        self.reldTdp = self.dTdp*self.paramset/self.y0[-1]
        
    def calcdSdp(self, res=50):
        """ Function to calculate the sensitivity state profile shapes
        to parameter perturbations, from wilkins. Might want to move
        this to a new class. """

        ts = np.linspace(0, self.y0[-1], res, endpoint=True)

        integrator = cs.CVodesIntegrator(self.model)

        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", self.y0[-1])
        integrator.setOption("numeric_jacobian", True)
        integrator.setOption("number_of_fwd_dir", self.NP)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol",
                             self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        integrator.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        integrator.setInput(self.paramset,cs.INTEGRATOR_P)

        p0_seed = np.eye(self.NP)
        for i in range(0,self.NP):
            integrator.setFwdSeed(p0_seed[i],cs.INTEGRATOR_P,i)
        
        for i in range(0,self.NP):
            integrator.setFwdSeed(self.S0[:,i],cs.INTEGRATOR_X0,i)

        sim = cs.Simulator(integrator, ts)
        sim.setOption("number_of_fwd_dir", self.NP)
        # sim.setOption("fsens_err_con", 1)
        # sim.setOption("fsens_abstol", self.intoptions['sensabstol'])
        # sim.setOption("fsens_reltol", self.intoptions['sensreltol'])
        sim.init()
        sim.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        sim.setInput(self.paramset,cs.INTEGRATOR_P)

        p0_seed = np.eye(self.NP)
        for i in range(0,self.NP):
            sim.setFwdSeed(p0_seed[i],cs.INTEGRATOR_P,i)
        
        for i in range(0,self.NP):
            sim.setFwdSeed(self.S0[:,i],cs.INTEGRATOR_X0,i)

        sim.evaluate(nfdir=self.NP)
        
        # Raw sensitivity matrix S, calculated with initial conditions
        # S_0 = Z[0]. This matrix is not periodic, and will grow
        # unbounded with time
        S = np.array([sim.fwdSens(cs.INTEGRATOR_X0, i).toArray() for i in
                      xrange(self.NP)])
        S = S.swapaxes(0,1).swapaxes(1,2) # S[t,y,p]
        y = sim.output().toArray()

        # Periodic Z matrix, defined as the state sensitivity with
        # constant period (Wilkins 2009, page 2710)
        Z = np.zeros(S.shape)
        for i in xrange(res):
            Z[i] = S[i] + (ts[i]/self.y0[-1])*np.outer(self.dydt(y[i]),
                                                       self.dTdp)

        self.Z = Z

        # Periodic W matrix, a projection of the Z matrix. Captures the
        # change in amplitude of the state trajectory without taking
        # into account changes in phase.
        W = np.zeros(S.shape)
        for i, (y_i, Z_i) in enumerate(zip(y,Z)):
            W[i] = (np.eye(self.NEQ) - 
                    np.outer(y_i, y_i)/np.linalg.norm(y_i)**2).dot(Z_i)

        self.W = W


    def findPRC(self, res=100, num_cycles=20):
        """ Function to calculate the phase response curve with
        specified resolution """

        # Make sure the lc object exists
        if not hasattr(self, 'lc'): self.limitCycle()

        # Get a state that is not at a local max/min (0 should be at
        # max)
        state_ind = 1
        while self.dydt(self.y0[:-1])[state_ind] < 1E-5: state_ind += 1

        integrator = cs.CVodesIntegrator(self.model)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", num_cycles*self.y0[-1])
        integrator.setOption("numeric_jacobian", True)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        seed = np.zeros(self.NEQ)
        seed[state_ind] = 1.
        integrator.setInput(self.y0[:-1], cs.INTEGRATOR_X0)
        integrator.setInput(self.paramset, cs.INTEGRATOR_P)
        integrator.setAdjSeed(seed, cs.INTEGRATOR_XF)
        integrator.evaluate(0, 1)
        adjsens = integrator.adjSens(cs.INTEGRATOR_X0).toArray().flatten()

        from scipy.integrate import odeint

        def adj_func(y, t):
            """ t will increase, trace limit cycle backwards through -t. y
            is the vector of adjoint sensitivities """
            jac = self.dfdy(self.lc((-t)%self.y0[-1]))
            return y.dot(jac)

        seed = adjsens

        self.prc_ts = np.linspace(0, self.y0[-1], res)
        P = odeint(adj_func, seed, self.prc_ts)[::-1] # Adjoint matrix
        self.sPRC = self._t_to_phi(P/self.dydt(self.y0[:-1])[state_ind])
        dfdp = np.array([self.dfdp(self.lc(t)) for t in self.prc_ts])
        # Must rescale f to \hat{f}, inverse of rescaling t
        self.pPRC = np.array([self.sPRC[i].dot(self._phi_to_t(dfdp[i]))
                              for i in xrange(len(self.sPRC))])
        self.rel_pPRC = self.pPRC*np.array(self.paramset)

        # Create interpolation object for the state phase response curve
        self.sPRC_interp = self.interp_sol(self.prc_ts, self.sPRC)
        





        # ts = np.linspace(0*self.y0[-1], 5*self.y0[-1], num=res)
        # integrator = cs.CVodesIntegrator(self.modlT)
        # integrator.setOption("abstol", self.intoptions['sensabstol'])
        # integrator.setOption("reltol", self.intoptions['sensreltol'])
        # integrator.setOption("max_num_steps",
        #                      self.intoptions['sensmaxnumsteps'])
        # integrator.setOption("sensitivity_method",
        #                      self.intoptions['sensmethod']);
        # integrator.setOption("t0", 0)
        # integrator.setOption("tf", 1)
        # integrator.setOption("numeric_jacobian", True)
        # integrator.setOption("fsens_err_con", 1)
        # integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        # integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        # integrator.init()

        # seed = [1] + [0]*(self.NEQ-1)
        # y0 = self.y0[:-1]

        # integrator.setInput(y0, cs.INTEGRATOR_X0)

        # adjoint_sensitivities = []
        # ys = []
        # dfdp = []
        # param = list(self.paramset) + [ts[0]]

        # for param[-1] in ts:       
        #     integrator.setInput(param, cs.INTEGRATOR_P)
        #     integrator.setAdjSeed(seed, cs.INTEGRATOR_XF)
        #     integrator.evaluate(0, 1)
        #     adjsens = integrator.adjSens(cs.INTEGRATOR_X0).toArray().flatten()
        #     adjoint_sensitivities.append(adjsens)   
        #     ys.append(integrator.output().toArray().flatten())
        #     dfdp.append(self.dfdp(ys[-1]))


        # Q = np.array(adjoint_sensitivities)
        # dfdp = np.array(dfdp)
        # self.pPRC = np.array([Q[i].dot(dfdp[i]) for i in xrange(len(Q))])
        # self.relpPRC = self.pPRC*np.array(self.paramset)


    def _create_ARC_model(self, numstates=1):
        """ Create model with quadrature for amplitude sensitivities
        numstates might allow us to calculate entire sARC at once, but
        now will use seed method. """

        # Allocate symbolic vectors for the model
        dphidt = cs.ssym('dphidt', numstates)
        t      = self.model.inputSX(cs.DAE_T)    # time
        xd     = self.model.inputSX(cs.DAE_X)    # differential state
        s      = cs.ssym("s", self.NEQ, numstates) # sensitivities
        p      = cs.vertcat([self.model.inputSX(2), dphidt]) # parameters

        # Symbolic function (from input model)
        ode_rhs = self.model.outputSX()

        # symbolic jacobians
        jac_x = self.model.jac(cs.DAE_X, cs.DAE_X)   
        sens_rhs = jac_x.mul(s)

        quad = cs.ssym('q', self.NEQ, numstates)
        for i in xrange(numstates):
            quad[:,i] = 2*(s[:,i] - dphidt*ode_rhs)*(xd - self.avg)

        shape = (self.NEQ*numstates, 1)

        x = cs.vertcat([xd, s.reshape(shape)])
        ode = cs.vertcat([ode_rhs, sens_rhs.reshape(shape)])
        ffcn = cs.SXFunction(cs.daeIn(t=t, x=x, p=p),
                             cs.daeOut(ode=ode, quad=quad))
        return ffcn


    def _sarc_single_time(self, time, seed):
        """ Calculate the state amplitude response to an infinitesimal
        perturbation in the direction of seed, at specified time. """

        # Initialize model and sensitivity states
        x0 = np.zeros(2*self.NEQ)
        x0[:self.NEQ] = self.lc(time)
        x0[self.NEQ:] = seed
        
        # Add dphi/dt from seed perturbation
        param = np.zeros(self.NP + 1)
        param[:self.NP] = self.paramset
        param[-1] = self.sPRC_interp(time).dot(seed)

        # Evaluate model
        self.sarc_int.setInput(x0, cs.INTEGRATOR_X0)
        self.sarc_int.setInput(param, cs.INTEGRATOR_P)
        self.sarc_int.evaluate()
        amp_change = self.sarc_int.output(cs.INTEGRATOR_QF).toArray()
        self.sarc_int.reset()

        amp_change *= (2*np.pi)/(self.y0[-1])

        return amp_change


    def _findARC_seed(self, seeds, res=100, trans=3): 

        # Calculate necessary quantities
        if not hasattr(self, 'avg'): self.average()
        if not hasattr(self, 'sPRC'): self.findPRC(res)

        # Set up quadrature integrator
        self.sarc_int = cs.CVodesIntegrator(self._create_ARC_model())
        self.sarc_int.setOption("abstol", self.intoptions['sensabstol'])
        self.sarc_int.setOption("reltol", self.intoptions['sensreltol'])
        self.sarc_int.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        self.sarc_int.setOption("t0", 0)
        self.sarc_int.setOption("tf", trans*self.y0[-1])
        self.sarc_int.setOption("numeric_jacobian", True)

        self.sarc_int.init()

        t_arc = np.linspace(0, self.y0[-1], res)
        arc = np.array([self._sarc_single_time(t, seed) for t, seed in
                        zip(t_arc, seeds)]).squeeze()
        return t_arc, arc

    def findSARC(self, state, res=100, trans=3):
        """ Find amplitude response curve from pertubation to state """
        seed = np.zeros(self.NEQ)
        seed[state] = 1.
        return self._findARC_seed([seed]*res, res, trans)

    def findPARC(self, param, res=100, trans=3, rel=False):
        """ Find ARC from temporary perturbation to parameter value """
        t_arc = np.linspace(0, self.y0[-1], res)
        dfdp = self._phi_to_t(self.dfdp(self.lc(t_arc))[:,:,param])
        t_arc, arc = self._findARC_seed(dfdp, res, trans)
        if rel: arc *= self.paramset[param]/self.avg
        return t_arc, arc

    def findARC_whole(self, res=100, trans=3):
        """ Calculate entire sARC matrix, which will be faster than
        calcualting for each parameter """

        # Calculate necessary quantities
        if not hasattr(self, 'avg'): self.average()
        if not hasattr(self, 'sPRC'): self.findPRC(res)

        # Set up quadrature integrator
        self.sarc_int = cs.CVodesIntegrator(
            self._create_ARC_model(numstates=self.NEQ))
        self.sarc_int.setOption("abstol", self.intoptions['sensabstol'])
        self.sarc_int.setOption("reltol", self.intoptions['sensreltol'])
        self.sarc_int.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        self.sarc_int.setOption("t0", 0)
        self.sarc_int.setOption("tf", trans*self.y0[-1])
        self.sarc_int.setOption("numeric_jacobian", True)
        self.sarc_int.init()

        self.arc_ts = np.linspace(0, self.y0[-1], res)
        
        amp_change = []
        for t in self.arc_ts:
            # Initialize model and sensitivity states
            x0 = np.zeros(self.NEQ*(self.NEQ + 1))
            x0[:self.NEQ] = self.lc(t)
            x0[self.NEQ:] = np.eye(self.NEQ).flatten()
            
            # Add dphi/dt from seed perturbation
            param = np.zeros(self.NP + self.NEQ)
            param[:self.NP] = self.paramset
            param[self.NP:] = self.sPRC_interp(t)

            # Evaluate model
            self.sarc_int.setInput(x0, cs.INTEGRATOR_X0)
            self.sarc_int.setInput(param, cs.INTEGRATOR_P)
            self.sarc_int.evaluate()
            out = self.sarc_int.output(cs.INTEGRATOR_QF).toArray()
            amp_change += [out*2*np.pi/self.y0[-1]]
                                        

        #[time, state_out, state_in]
        self.sARC = np.array(amp_change)
        dfdp = np.array([self.dfdp(self.lc(t)) for t in self.arc_ts])
        self.pARC = np.array([self.sARC[i].dot(self._phi_to_t(dfdp[i]))
                              for i in xrange(len(self.sARC))])

        self.rel_pARC = (np.array(self.paramset) * self.pARC /
                         np.atleast_2d(self.avg).T)
                       

    def remove_unpickleable_objects(self):
        """
        Iterate over attributes in the class, removing all casadi or
        Periodic instances. 
        """
        self.removed_attrs = []
        for attr in dir(self):
            if ('casadi' in str(type(getattr(self, attr))) or 'Periodic'
                in str(type(getattr(self, attr)))):
                delattr(self, attr)
                self.removed_attrs += [attr]

        delattr(self, 'lc')
        self.removed_attrs += ['lc']


    def _simulate(self, ts, y0=None, paramset=None):
        """ Simulate the class, outputing the solution at the times
        specified by ts. Optional inputs of y0 and paramsets to use ones
        other than those currently in the class """

        if y0 is None: y0 = self.y0[:-1]
        if paramset is None: paramset = self.paramset

        int = cs.CVodesIntegrator(self.model)
        int.setOption("abstol"       , self.intoptions['lc_abstol'])
        int.setOption("reltol"       , self.intoptions['lc_reltol'])
        int.setOption("max_num_steps", self.intoptions['lc_maxnumsteps'])
        int.setOption("tf"           , self.y0[-1])

        sim = cs.Simulator(int, ts)
        sim.init()
        
        # Input Arguments
        sim.setInput(y0, cs.INTEGRATOR_X0)
        sim.setInput(paramset, cs.INTEGRATOR_P)
        sim.evaluate()
        return sim.output().toArray()

    def phase_of_point(self, point, error=False, tol=1E-3):
        """ Finds the phase at which the distance from the point to the
        limit cycle is minimized. phi=0 corresponds to the definition of
        y0, returns the phase and the minimum distance to the limit
        cycle """

        point = np.asarray(point)
        for i in xrange(100):
            dist = cs.ssym("dist")
            x = self.model.inputSX(cs.DAE_X)
            ode = self.model.outputSX()
            dist_ode = cs.sumAll(2*(x - point)*ode)

            cat_x   = cs.vertcat([x, dist])
            cat_ode = cs.vertcat([ode, dist_ode])

            dist_model = cs.SXFunction(
                cs.daeIn(t=self.model.inputSX(cs.DAE_T), x=cat_x,
                         p=self.model.inputSX(cs.DAE_P)),
                cs.daeOut(ode=cat_ode))

            dist_model.setOption("name","distance model")

            dist_0 = ((self.y0[:-1] - point)**2).sum()
            cat_y0 = np.hstack([self.y0[:-1], dist_0, self.y0[-1]])

            roots_class = pBase(dist_model, self.paramset, cat_y0)
            # roots_class.solveBVP()
            roots_class.roots()

            phase = self._t_to_phi(roots_class.Tmin[-1])
            distance = roots_class.Ymin[-1]

            if distance < tol: return phase, distance

            intr = cs.CVodesIntegrator(self.model)
            intr.setOption("abstol", self.intoptions['transabstol'])
            intr.setOption("reltol", self.intoptions['transreltol'])
            intr.setOption("tf", self.y0[-1])
            intr.setOption("max_num_steps",
                           self.intoptions['transmaxnumsteps'])
            intr.setOption("disable_internal_warnings", True)
            intr.init()
            intr.setInput(point, cs.INTEGRATOR_X0)
            intr.setInput(self.paramset, cs.INTEGRATOR_P)
            intr.evaluate()
            point = intr.output().toArray().flatten()

        raise RuntimeError("Point failed to converge to limit cycle")
        




if __name__ == "__main__":
    from CommonFiles.Models.leloup16model import model, paramset, NEQ

    new = pBase(model(), paramset, np.ones(NEQ+1))
    new.intPastTrans(300)

    class laptimer:
        def __init__(self):
            self.time = time()

        def __call__(self):
            ret = time() - self.time
            self.time = time()
            return ret

        def __str__(self):
            return "%.3E"%self()

        def __repr__(self):
            return "%.3E"%self()

    lap = laptimer()

    y0 = np.array(list(new.y0))
    new.approxY0(tol=1E-4)
    print "findy0_periodic:\t%.3E\t"%lap(), new.y0[-1]

    y0 = np.array(list(new.y0))
    new.solveBVP_periodic()
    print "mine:\t%.3E\t"%lap(), new.y0[-1]
    
    new.y0 = y0
    new.solveBVP_casadi()
    print "casadi:\t%.3E\t"%lap(), new.y0[-1]

    new.y0 = y0
    new.solveBVP_scipy()
    print "scipy:\t%.3E\t"%lap(), new.y0[-1]

    new.limitCycle()
    new.roots()
    new.average()
    new.offset_t(1.)

    new.first_order_sensitivity()

    new.findPRC()

    leloup16modelsens = np.array([
        -1.53952723e+01,  -7.23860215e-01,  -3.13928640e+00,
        -4.31151516e-01,  -4.81132412e-01,   1.61381708e+00,
        -9.46384702e+00,   4.05026442e+00,  -8.82800527e+00,
         8.41957544e-01,   3.03913035e+00,  -2.40513043e+02,
         1.38989320e+00,  -6.16960105e+01,  -8.46879599e+01,
         1.32602113e+00,   1.19444124e+00,  -6.75604744e-01,
        -8.41185747e+00,   2.36739757e+00,  -1.64073926e+00,
         2.61482062e+00,   2.22899736e+01,  -1.49389552e+00,
        -1.67779410e+00,   1.78909803e+00,   2.57987329e-03,
        -3.08727916e+00,   4.72990766e-01,  -1.00548152e-01,
        -2.37954223e+00,   2.74994825e+00,  -4.18359410e-01,
         6.64670890e-02,   1.88278515e+00,  -4.64566853e+00,
        -3.24910052e+00,   3.57359345e+00,   2.45614230e+00,
        -1.60680729e-01,  -4.45067923e-01,   2.53210923e-02,
        -7.51955982e-01,  -5.92901913e-02,  -1.81258931e-01,
        -2.21523951e-01,  -2.66388035e+01,   1.84308252e+00,
        -1.37787867e+01,   2.30388798e+01,  -3.25183504e+00,
         8.89292457e+00])

    print "Sens Error:\t%0.3E"%np.linalg.norm(new.dTdp-leloup16modelsens)

