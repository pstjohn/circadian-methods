# COMMON IMPORTS
from itertools import izip
import numpy  as np
from matplotlib.ticker import MaxNLocator

# THIRD PARTY IMPORTS
import casadi as cs

# MY IMPORTS
from Utilities import fnlist, get_ind, spline
from Utilities import colors as colors_default



class LimitCycleCollocation:
    """
    Class to solve for a parameterization of a given model which
    minimizes the distance between x(t) and given data points. 
    """

    def __init__(self, model, paramset):

        self.paramset = paramset + [24.0]
        # Define some useful constants

        self.NLPdata = {
            'TF' : 1,
            'NK' : 10,
            'DEG' : 5,
            'TMAX' : 100,
            'TMIN' : 1,
            'XMAX' : 1E+2,
            'XMIN' : 1E-3,
            'PRC' : False}


        # Check for machine-specific linear solver designation,
        # specified in linear_solver.p
        try:
            f = open('linear_solver.p')
            linear_solver = f.readline()[:-1]
        except Exception: linear_solver = 'ma57'

        # Collocation Solver Options.
        # These options are all passed directly to the CasADi IPOPT
        # solver class, and as such must match the documentation from
        # CasADi
        self.IpoptOpts = {
            'expand_f'                   : True,
            'expand_g'                   : True,
            'generate_hessian'           : True,
            'max_iter'                   : 6000,
            'tol'                        : 1e-6,
            'acceptable_constr_viol_tol' : 1E-4,
            'linear_solver'              : linear_solver,
            'expect_infeasible_problem'  : "yes",
            #'print_level'               : 10      # For Debugging
        }

        if model:
            self.AttachModel(model)

        self.NP = 1
        self.TF = 1.
        self.DEG = 5
        self.NK = 10
        self.NICP = 1
        self.NMP = 0

    def CoefficentSetup(self):
        """
        Creates and returns coefficients for the interpolating
        polynomials. Also returns and stores the grid for all XD points
        in self.tgrid 

        Requirements: None (set up additional options through direct
        calls to self.NLPdata and self.IpoptOpts, PARMAX, PARMIN, XMAX,
        MIN, etc.
        """

        # Legendre collocation points
        legendre_points1 = [0, 0.500000]
        legendre_points2 = [0, 0.211325, 0.788675]
        legendre_points3 = [0, 0.112702, 0.500000, 0.887298]
        legendre_points4 = [0, 0.069432, 0.330009, 0.669991, 0.930568]
        legendre_points5 = [0, 0.046910, 0.230765, 0.500000, 0.769235,
                            0.953090]
        legendre_points  = [0, legendre_points1, legendre_points2,
                            legendre_points3, legendre_points4,
                            legendre_points5]

        # Radau collocation points
        radau_points1 = [0, 1.000000]
        radau_points2 = [0, 0.333333, 1.000000]
        radau_points3 = [0, 0.155051, 0.644949, 1.000000]
        radau_points4 = [0, 0.088588, 0.409467, 0.787659, 1.000000]
        radau_points5 = [0, 0.057104, 0.276843, 0.583590, 0.860240,
                         1.000000]
        radau_points  = [0, radau_points1, radau_points2, radau_points3,
                         radau_points4, radau_points5]

        # Type of collocation points
        # LEGENDRE = 0
        RADAU = 1
        collocation_points = [legendre_points, radau_points]
        self.NLPdata['collocation_points'] = collocation_points

        # Radau collocation points
        self.NLPdata['cp'] = cp = RADAU
        # Size of the finite elements
        self.h = self.NLPdata['TF']/self.NLPdata['NK']

        # Coefficients of the collocation equation
        C = np.zeros((self.NLPdata['DEG']+1,self.NLPdata['DEG']+1))
        # Coefficients of the continuity equation
        D = np.zeros(self.NLPdata['DEG']+1)
        # Coefficients for integration
        E = np.zeros(self.NLPdata['DEG']+1)

        # Collocation point
        tau = cs.ssym("tau")
          
        # All collocation time points
        tau_root = collocation_points[cp][self.NLPdata['DEG']]
        T = np.zeros((self.NLPdata['NK'],self.NLPdata['DEG']+1))
        for i in range(self.NLPdata['NK']):
          for j in range(self.NLPdata['DEG']+1):
                T[i][j] = self.h*(i + tau_root[j])

        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at
        # the collocation point
        for j in range(self.NLPdata['DEG']+1):
            L = 1
            for j2 in range(self.NLPdata['DEG']+1):
                if j2 != j:
                    L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
            lfcn = cs.SXFunction([tau],[L])
            lfcn.init()
            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation
            lfcn.setInput(1.0)
            lfcn.evaluate()
            D[j] = lfcn.output()

            # Evaluate the time derivative of the polynomial at all
            # collocation points to get the coefficients of the
            # continuity equation
            for j2 in range(self.NLPdata['DEG']+1):
                lfcn.setInput(tau_root[j2])
                lfcn.setFwdSeed(1.0)
                lfcn.evaluate(1,0)
                C[j][j2] = lfcn.fwdSens()

            # lint = cs.CVodesIntegrator(lfcn)

            tg = np.array(tau_root)*self.h
            for k in range(self.NLPdata['NK']):
                if k == 0:
                    tgrid = tg
                else:
                    tgrid = np.append(tgrid,tgrid[-1]+tg)

        self.tgrid = tgrid
        self.C = C
        self.D = D
        # weights for the integration of function along lagrange
        # polynomial
        self.E = [0., 0.118463, 0.239314, 0.284445, 0.239314, 0.118463]

        # Set up XMAX and XMIN variables. Check if they are a
        # vector, if not resize to number of state variables
        try:
            assert len(self.NLPdata['XMAX']) == len(self.NLPdata['XMIN']) == self.NEQ, \
            "State variable bounds not correct length"
        except TypeError:
            self.NLPdata['XMAX'] = [self.NLPdata['XMAX']] * self.NEQ
            self.NLPdata['XMIN'] = [self.NLPdata['XMIN']] * self.NEQ


    def AttachModel(self, model):
        """
        Attach a casadi ODE model. Model should be re-formulated at a
        later time when desired sensitivities have been specified. 
        Algebraic states are a remnent of the dae_collocation.py example
        file, and should not be expected to work.
        """

        self.model = model; self.model.init()
        self.NEQ = self.model.input(cs.DAE_X).size()
        self.NP  = self.model.input(cs.DAE_P).size()

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

    def ReformModel(self):
        """
        Reform self.model to conform to the standard (implicit) form
        used by the rest of the collocation calculations. Should also
        append sensitivity states (dy/dp_i) given as a list to the end
        of the ode model.

        Monodromy - calculate a symbolic monodromy matrix by integrating
        dy/dy_0,i 
        """
        
        T     = cs.ssym("T")                 # Period
        
        if self.NLPdata['PRC']:
            self.NVAR = 2*self.NEQ
            s = cs.ssym("s", self.NEQ, 1)    # sensitivities
            jac_x = self.model.jac(cs.DAE_X, cs.DAE_X)   
            prc_rhs = T*jac_x.T.mul(s)
            
        else:
            self.NVAR = self.NEQ
            s = cs.ssym("s", 0, 1)           # sensitivities
            prc_rhs = 0

        # Allocate symbolic vectors for the model
        t     = self.model.inputSX(cs.DAE_T) # time
        u     = cs.ssym("u", 0, 1)           # control (empty)
        xd_in = self.model.inputSX(cs.DAE_X) # differential state
        xa    = cs.ssym("xa", 0, 1)          # algebraic state (empty)
        xddot = cs.ssym("xd", self.NVAR)     # differential state dt
        p     = self.model.inputSX(2)        # parameters

        # Symbolic function (from input model)
        ode_rhs = T*self.model.outputSX()

        # symbolic jacobians

        ode = xddot[:self.NEQ] - ode_rhs
        sens = xddot[self.NEQ:] - prc_rhs
        xd = cs.vertcat([xd_in, s])
        tot = cs.vertcat([ode, sens])
        p_in = cs.vertcat([p,T])
        
        self.rfmod = cs.SXFunction([t,xddot,xd,xa,u,p_in], [tot])

        self.rfmod.init()



    def Initialize(self):
        """
        Uses the finished p_init and x_init values to finalize the
        NLPd structure with full matricies on state bounds, parameter
        bounds, etc.

        Requirements: AttachModel, AttachData, SetInitialValues,
        (ParameterEstimation optional)
        """

        NLPd = self.NLPdata
        NSENS = self.NEQ if self.NLPdata['PRC'] else 0
            
        if type(self.NLPdata['XMAX']) is np.ndarray: self.NLPdata['XMAX'] = self.NLPdata['XMAX'].tolist()
        if type(self.NLPdata['XMIN']) is np.ndarray: self.NLPdata['XMIN'] = self.NLPdata['XMIN'].tolist()

        # Algebraic state bounds and initial guess
        NLPd['xA_min']  = np.array([])
        NLPd['xA_max']  = np.array([])
        NLPd['xAi_min'] = np.array([])
        NLPd['xAi_max'] = np.array([])
        NLPd['xAf_min'] = np.array([])
        NLPd['xAf_max'] = np.array([])
        NLPd['xA_init'] = np.array((self.NLPdata['NK']*(self.NLPdata['DEG']+1))*[[]])

        # Control bounds
        NLPd['u_min']  = np.array([])
        NLPd['u_max']  = np.array([])
        NLPd['u_max']  = np.array([])
        NLPd['u_init'] = np.array((self.NLPdata['NK']*(self.NLPdata['DEG']+1))*[[]])
        
        # Differential state bounds and initial guess
        NLPd['xD_min']  =  np.array(self.NLPdata['XMIN'] + [-1E6]*NSENS)
        NLPd['xD_max']  =  np.array(self.NLPdata['XMAX'] + [+1E6]*NSENS)
        NLPd['xDi_min'] =  np.array(self.NLPdata['XMIN'] + [-1E6]*NSENS)
        NLPd['xDi_max'] =  np.array(self.NLPdata['XMAX'] + [+1E6]*NSENS)
        NLPd['xDf_min'] =  np.array(self.NLPdata['XMIN'] + [-1E6]*NSENS)
        NLPd['xDf_max'] =  np.array(self.NLPdata['XMAX'] + [+1E6]*NSENS)
        NLPd['xD_init'] =  np.ones((len(self.tgrid), self.NEQ + NSENS))
        # needs to be specified for every time interval
        

        # Parameter bounds and initial guess
        NLPd['p_min']  = self.NLPdata['TMIN']
        NLPd['p_max']  = self.NLPdata['TMAX']
        NLPd['p_init'] = 24.

    def CollocationSetup(self, warmstart=False):
        """
        Sets up NLP for collocation solution. Constructs initial guess
        arrays, constructs constraint and objective functions, and
        otherwise passes arguments to the correct places. This looks
        really inefficient and is likely unneccessary to run multiple
        times for repeated runs with new data. Not sure how much time it
        takes compared to the NLP solution.

        Run immediately before CollocationSolve.
        """
    
        # Dimensions of the problem
        nx    = self.NVAR # total number of states
        ndiff = nx        # number of differential states
        nalg  = 0         # number of algebraic states
        nu    = 0         # number of controls

        # Collocated variables
        NXD = self.NICP*self.NK*(self.DEG+1)*ndiff # differential states 
        NXA = self.NICP*self.NK*self.DEG*nalg      # algebraic states
        NU  = self.NK*nu                           # Parametrized controls
        NV  = NXD+NXA+NU+self.NP+self.NMP # Total variables
        self.NV = NV

        # NLP variable vector
        V = cs.msym("V",NV)
          
        # All variables with bounds and initial guess
        vars_lb   = np.zeros(NV)
        vars_ub   = np.zeros(NV)
        vars_init = np.zeros(NV)
        offset    = 0

        #
        # Split NLP vector into useable slices
        #
        # Get the parameters
        P = V[offset:offset+self.NP]
        vars_init[offset:offset+self.NP] = self.NLPdata['p_init']
        vars_lb[offset:offset+self.NP]   = self.NLPdata['p_min']
        vars_ub[offset:offset+self.NP]   = self.NLPdata['p_max']


        offset += self.NP # indexing variable

        # Get collocated states and parametrized control
        XD = np.resize(np.array([], dtype=cs.MX), (self.NK, self.NICP,
                                                   self.DEG+1)) 
        # NB: same name as above
        XA = np.resize(np.array([],dtype=cs.MX),(self.NK,self.NICP,self.DEG)) 
        # NB: same name as above
        U = np.resize(np.array([],dtype=cs.MX),self.NK)

        # Prepare the starting data matrix vars_init, vars_ub, and
        # vars_lb, by looping over finite elements, states, etc. Also
        # groups the variables in the large unknown vector V into XD and
        # XA(unused) for later indexing
        for k in range(self.NK):  
            # Collocated states
            for i in range(self.NICP):
                #
                for j in range(self.DEG+1):
                              
                    # Get the expression for the state vector
                    XD[k][i][j] = V[offset:offset+ndiff]
                    if j !=0:
                        XA[k][i][j-1] = V[offset+ndiff:offset+ndiff+nalg]
                    # Add the initial condition
                    index = (self.DEG+1)*(self.NICP*k+i) + j
                    if k==0 and j==0 and i==0:
                        vars_init[offset:offset+ndiff] = \
                            self.NLPdata['xD_init'][index,:]
                        
                        vars_lb[offset:offset+ndiff] = \
                                self.NLPdata['xDi_min']
                        vars_ub[offset:offset+ndiff] = \
                                self.NLPdata['xDi_max']
                        offset += ndiff
                    else:
                        if j!=0:
                            vars_init[offset:offset+nx] = \
                            np.append(self.NLPdata['xD_init'][index,:],
                                      self.NLPdata['xA_init'][index,:])
                            
                            vars_lb[offset:offset+nx] = \
                            np.append(self.NLPdata['xD_min'],
                                      self.NLPdata['xA_min'])

                            vars_ub[offset:offset+nx] = \
                            np.append(self.NLPdata['xD_max'],
                                      self.NLPdata['xA_max'])

                            offset += nx
                        else:
                            vars_init[offset:offset+ndiff] = \
                                    self.NLPdata['xD_init'][index,:]

                            vars_lb[offset:offset+ndiff] = \
                                    self.NLPdata['xD_min']

                            vars_ub[offset:offset+ndiff] = \
                                    self.NLPdata['xD_max']

                            offset += ndiff
            
            # Parametrized controls (unused here)
            U[k] = V[offset:offset+nu]

        # Attach these initial conditions to external dictionary
        self.NLPdata['v_init'] = vars_init
        self.NLPdata['v_ub'] = vars_ub
        self.NLPdata['v_lb'] = vars_lb

        # Setting up the constraint function for the NLP. Over each
        # collocated state, ensure continuitity and system dynamics
        g = []
        lbg = []
        ubg = []

        # For all finite elements
        for k in range(self.NK):
            for i in range(self.NICP):
                # For all collocation points
                for j in range(1,self.DEG+1):   		
                    # Get an expression for the state derivative
                    # at the collocation point
                    xp_jk = 0
                    for j2 in range (self.DEG+1):
                        # get the time derivative of the differential
                        # states (eq 10.19b)
                        xp_jk += self.C[j2][j]*XD[k][i][j2]
                    
                    # Add collocation equations to the NLP
                    [fk] = self.rfmod.call([0., xp_jk/self.h,
                                            XD[k][i][j], XA[k][i][j-1],
                                            U[k], P])
                    
                    # impose system dynamics (for the differential
                    # states (eq 10.19b))
                    g += [fk[:ndiff]]
                    lbg.append(np.zeros(ndiff)) # equality constraints
                    ubg.append(np.zeros(ndiff)) # equality constraints

                    # impose system dynamics (for the algebraic states
                    # (eq 10.19b)) (unused)
                    g += [fk[ndiff:]]                               
                    lbg.append(np.zeros(nalg)) # equality constraints
                    ubg.append(np.zeros(nalg)) # equality constraints
                    
                # Get an expression for the state at the end of the finite
                # element
                xf_k = 0
                for j in range(self.DEG+1):
                    xf_k += self.D[j]*XD[k][i][j]
                    
                # if i==self.NICP-1:

                # Add continuity equation to NLP
                if k+1 != self.NK: # End = Beginning of next
                    g += [XD[k+1][0][0] - xf_k]
                    lbg.append(-self.NLPdata['CONtol']*np.ones(ndiff))
                    ubg.append(self.NLPdata['CONtol']*np.ones(ndiff))
                
                else: # At the last segment
                    # Periodicity constraints (only for NEQ)
                    g += [XD[0][0][0][:self.NEQ] - xf_k[:self.NEQ]]
                    lbg.append(-self.NLPdata['CONtol']*np.ones(self.NEQ))
                    ubg.append(self.NLPdata['CONtol']*np.ones(self.NEQ))


                # else:
                #     g += [XD[k][i+1][0] - xf_k]
                
        # Flatten contraint arrays for last addition
        lbg = np.concatenate(lbg).tolist()
        ubg = np.concatenate(ubg).tolist()

        # Constraint to protect against fixed point solutions
        if self.NLPdata['FPgaurd'] is True:
            fout = self.model.call(cs.daeIn(t=self.tgrid[0],
                                            x=XD[0,0,0][:self.NEQ],
                                               p=V[:self.NP]))[0]
            g += [cs.MX(cs.sumAll(fout**2))]
            lbg.append(np.array(self.NLPdata['FPTOL']))
            ubg.append(np.array(cs.inf))

        elif self.NLPdata['FPgaurd'] is 'all':
            fout = self.model.call(cs.daeIn(t=self.tgrid[0],
                                            x=XD[0,0,0][:self.NEQ],
                                               p=V[:self.NP]))[0]
            g += [cs.MX(fout**2)]
            lbg += [self.NLPdata['FPTOL']]*self.NEQ
            ubg += [cs.inf]*self.NEQ



        # Nonlinear constraint function
        gfcn = cs.MXFunction([V],[cs.vertcat(g)])

        # Minimize derivative of first state variable at t=0
        xp_0 = 0
        for j in range (self.NLPdata['DEG']+1):
            # get the time derivative of the differential
            # states (eq 10.19b)
            xp_0 += self.C[j][0]*XD[0][j][0]

        obj = xp_0
        ofcn = cs.MXFunction([V], [obj])

        self.CollocationSolver = cs.IpoptSolver(ofcn,gfcn)

        for opt,val in self.IpoptOpts.iteritems():
            self.CollocationSolver.setOption(opt,val)

        self.CollocationSolver.setOption('obj_scaling_factor',
                                         len(vars_init))
        
        if warmstart:
            self.CollocationSolver.setOption('warm_start_init_point','yes')
        
        # initialize the self.CollocationSolver
        self.CollocationSolver.init()
          
        # Initial condition
        self.CollocationSolver.setInput(vars_init,cs.NLP_X_INIT)

        # Bounds on x
        self.CollocationSolver.setInput(vars_lb,cs.NLP_LBX)
        self.CollocationSolver.setInput(vars_ub,cs.NLP_UBX)

        # Bounds on g
        self.CollocationSolver.setInput(np.array(lbg),cs.NLP_LBG)
        self.CollocationSolver.setInput(np.array(ubg),cs.NLP_UBG)

        if warmstart:
            self.CollocationSolver.setInput( \
                    self.WarmStartData['NLP_X_OPT'],cs.NLP_X_INIT)
            self.CollocationSolver.setInput( \
                    self.WarmStartData['NLP_LAMBDA_G'],cs.NLP_LAMBDA_INIT)
            self.CollocationSolver.setOutput( \
                    self.WarmStartData['NLP_LAMBDA_X'],cs.NLP_LAMBDA_X)


    # def CollocationSetup(self):
    #     """
    #     Sets up NLP for collocation solution. Constructs initial guess
    #     arrays, constructs constraint and objective functions, and
    #     otherwise passes arguments to the correct places. This looks
    #     really inefficient and is likely unneccessary to run multiple
    #     times for repeated runs with new data. Not sure how much time it
    #     takes compared to the NLP solution.

    #     Run immediately before CollocationSolve.
    #     """
    #     
    #     # Dimensions of the problem
    #     nx    = self.NVAR # total number of states
    #     ndiff = nx        # number of differential states
    #     nalg  = 0         # number of algebraic states
    #     nu    = 0         # number of controls

    #     # Collocated variables
    #     NXD = self.NLPdata['NK']*(self.NLPdata['DEG']+1)*ndiff # differential states 
    #     NXA = self.NLPdata['NK']*self.NLPdata['DEG']*nalg      # algebraic states
    #     NU  = self.NLPdata['NK']*nu                           # Parametrized controls
    #     NV  = NXD+NXA+NU+1 # Total variables
    #     self.NV = NV

    #     # NLP variable vector
    #     V = cs.msym("V",NV)
    #       
    #     # All variables with bounds and initial guess
    #     vars_lb   = np.zeros(NV)
    #     vars_ub   = np.zeros(NV)
    #     vars_init = np.zeros(NV)
    #     offset    = 0

    #     #
    #     # Split NLP vector into useable slices
    #     #
    #     # Get the period
    #     # T = V[0]
    #     # vars_init[0] = self.NLPdata['p_init']
    #     # vars_lb[0]   = self.NLPdata['p_min']
    #     # vars_ub[0]   = self.NLPdata['p_max']

    #     # offset += 1 # indexing variable

    #     # Get collocated states and parametrized control
    #     XD = np.resize(np.array([], dtype=cs.MX), (self.NLPdata['NK'], 
    #                                                self.NLPdata['DEG']+1)) 
    #     # NB: same name as above
    #     XA = np.resize(np.array([],dtype=cs.MX),(self.NLPdata['NK'], self.NLPdata['DEG'])) 
    #     # NB: same name as above
    #     U = np.resize(np.array([],dtype=cs.MX),self.NLPdata['NK'])

    #     # Prepare the starting data matrix vars_init, vars_ub, and
    #     # vars_lb, by looping over finite elements, states, etc. Also
    #     # groups the variables in the large unknown vector V into XD and
    #     # XA(unused) for later indexing
    #     for k in range(self.NLPdata['NK']):  
    #         # Collocated states
    #         for j in range(self.NLPdata['DEG']+1):
    #                       
    #             # Get the expression for the state vector
    #             XD[k][j] = V[offset:offset+ndiff]
    #             if j !=0:
    #                 XA[k][j-1] = V[offset+ndiff:offset+ndiff+nalg]
    #             # Add the initial condition
    #             index = (self.NLPdata['DEG']+1) + j
    #             if k==0 and j==0:
    #                 vars_init[offset:offset+ndiff] = \
    #                     self.NLPdata['xD_init'][index,:]
    #                 
    #                 vars_lb[offset:offset+ndiff] = \
    #                         self.NLPdata['xDi_min']
    #                 vars_ub[offset:offset+ndiff] = \
    #                         self.NLPdata['xDi_max']
    #                 offset += ndiff
    #             else:
    #                 if j!=0:
    #                     vars_init[offset:offset+nx] = \
    #                     np.append(self.NLPdata['xD_init'][index,:],
    #                               self.NLPdata['xA_init'][index,:])
    #                     
    #                     vars_lb[offset:offset+nx] = \
    #                     np.append(self.NLPdata['xD_min'],
    #                               self.NLPdata['xA_min'])

    #                     vars_ub[offset:offset+nx] = \
    #                     np.append(self.NLPdata['xD_max'],
    #                               self.NLPdata['xA_max'])

    #                     offset += nx
    #                 else:
    #                     vars_init[offset:offset+ndiff] = \
    #                             self.NLPdata['xD_init'][index,:]

    #                     vars_lb[offset:offset+ndiff] = \
    #                             self.NLPdata['xD_min']

    #                     vars_ub[offset:offset+ndiff] = \
    #                             self.NLPdata['xD_max']

    #                     offset += ndiff
    #         
    #         # Parametrized controls (unused here)
    #         U[k] = V[offset:offset+nu]

    #     # Attach these initial conditions to external dictionary
    #     self.NLPdata['v_init'] = vars_init
    #     self.NLPdata['v_ub'] = vars_ub
    #     self.NLPdata['v_lb'] = vars_lb

    #     # Setting up the constraint function for the NLP. Over each
    #     # collocated state, ensure continuitity and system dynamics
    #     g = []
    #     lbg = []
    #     ubg = []

    #     # For all finite elements
    #     for k in range(self.NLPdata['NK']):
    #         # For all collocation points
    #         for j in range(1,self.NLPdata['DEG']+1):   		
    #             # Get an expression for the state derivative
    #             # at the collocation point
    #             xp_jk = 0
    #             for j2 in range (self.NLPdata['DEG']+1):
    #                 # get the time derivative of the differential
    #                 # states (eq 10.19b)
    #                 xp_jk += self.C[j2][j]*XD[k][j2]
    #             
    #             # Add collocation equations to the NLP
    #             # P = cs.vertcat([np.array(self.paramset), T])
    #             P = cs.vertcat([np.array(self.paramset)])
    #             [fk] = self.rfmod.call([0., xp_jk/self.h,
    #                                     XD[k][j], P])
    #             
    #             # impose system dynamics (for the differential
    #             # states (eq 10.19b))
    #             g += [fk]
    #             lbg.append(np.zeros(ndiff)) # equality constraints
    #             ubg.append(np.zeros(ndiff)) # equality constraints
    #             
    #         # Get an expression for the state at the end of the finite
    #         # element
    #         xf_k = 0
    #         for j in range(self.NLPdata['DEG']+1):
    #             xf_k += self.D[j]*XD[k][j]
    #             
    #         # Add continuity equation to NLP
    #         if k+1 != self.NLPdata['NK']: # End = Beginning of next
    #             g += [XD[k+1][0] - xf_k]
    #             lbg.append(np.zeros(ndiff))
    #             ubg.append(np.zeros(ndiff))
    #         
    #         else: # At the last segment
    #             # Periodicity constraints (only for NEQ)
    #             g += [XD[0][0] - xf_k]
    #             lbg.append(np.zeros(ndiff))
    #             ubg.append(np.zeros(ndiff))

    #     # Flatten contraint arrays for last addition
    #     lbg = np.concatenate(lbg).tolist()
    #     ubg = np.concatenate(ubg).tolist()

    #     # Nonlinear constraint function
    #     gfcn = cs.MXFunction([V],[cs.vertcat(g)])

        # # Minimize derivative of first state variable at t=0
        # xp_0 = 0
        # for j in range (self.NLPdata['DEG']+1):
        #     # get the time derivative of the differential
        #     # states (eq 10.19b)
        #     xp_0 += self.C[j][0]*XD[0][j][0]

        # obj = xp_0
        # ofcn = cs.MXFunction([V], [obj])

    #     self.CollocationSolver = cs.IpoptSolver(ofcn,gfcn)

    #     for opt,val in self.IpoptOpts.iteritems():
    #         self.CollocationSolver.setOption(opt,val)

    #     self.CollocationSolver.setOption('obj_scaling_factor',
    #                                      len(vars_init))
    #     
    #     # initialize the self.CollocationSolver
    #     self.CollocationSolver.init()
    #       
    #     # Initial condition
    #     self.CollocationSolver.setInput(vars_init,cs.NLP_X_INIT)

    #     # Bounds on x
    #     self.CollocationSolver.setInput(vars_lb,cs.NLP_LBX)
    #     self.CollocationSolver.setInput(vars_ub,cs.NLP_UBX)

    #     # Bounds on g
    #     self.CollocationSolver.setInput(np.array(lbg),cs.NLP_LBG)
    #     self.CollocationSolver.setInput(np.array(ubg),cs.NLP_UBG)

    def CollocationSolve(self):
        self.CollocationSolver.solve()

        try: success = self.CollocationSolver.getStat('return_status')
        except Exception:
            raise RuntimeError('Collocation Solve Unsuccessful: No Flag')

        assert (success == 'Solve_Succeeded'),\
                "Collocation Solve Unsuccessful: " + str(success)

        # Retrieve the solution
        self.NLPdata['v_opt'] = v_opt = \
                np.array(self.CollocationSolver.output(cs.NLP_X_OPT))
        self.NLPdata['p_opt'] = v_opt[:self.NP].squeeze()
        x_opt = v_opt[self.NP:self.NV -
                      self.NMP].reshape(((self.NLPdata['DEG']+1)*self.NLPdata['NK'],
                                         self.NVAR))
        self.NLPdata['x_opt'] = x_opt[:,:self.NEQ]
        self.NLPdata['s_opt'] = \
        x_opt[:,self.NEQ:].reshape((len(self.tgrid), self.NSENS,
                                    self.NEQ))
        self.NLPdata['xd_opt'] = x_opt[1::(self.NLPdata['DEG']+1)]
        self.NLPdata['mp_opt'] = v_opt[self.NV-self.NMP:] 
        for i, entry in enumerate(self.mpars):
            if entry:
                self.Amatrix[i] *= \
                self.NLPdata['mp_opt'][sum(self.mpars[:i])]


        # Get data for warm restart
        self.WarmStartData={
           'NLP_X_OPT'    : self.CollocationSolver.output(cs.NLP_X_OPT),
           'NLP_COST'     : self.CollocationSolver.output(cs.NLP_COST),
           'NLP_LAMBDA_G' : self.CollocationSolver.output(cs.NLP_LAMBDA_G),
           'NLP_LAMBDA_X' : self.CollocationSolver.output(cs.NLP_LAMBDA_X)
           }

        return success



    def bvp_solution(self):
        """
        Generate and solve for the limit cycle numerically using the ODE
        and BVP procedures in pBase. Should return an error if the bvp
        solution is for some reason invalid. (Does not match collocation
        solution, not a limit cycle, etc)
        """

        assert (np.linalg.norm((self.NLPdata['x_opt'] -
                                self.NLPdata['x_opt'][0]).sum(0))) \
                > 1E-3, "Collocation solution non-oscillatory"


        self._create_limitcycle_class()

        if self.BvpOpts['findstationary']:
            assert self.limitcycle.findstationary() != 1, \
                "bvp_solution reached steady state (findstationary)"

        # Try to find initial conditions using approxY0, quietly print
        # error if failed
        try: self.limitcycle.approxY0(tol=self.BvpOpts['Y0TOL'])
        except (RuntimeWarning, RuntimeError) as ex:
            print ex
            pass

        # Errors shouldn't arise at this step, but they will. Better to
        # have steady-state or other informative error than an ambiguous
        # failed call to a bvp method (except stability calculation)
        stability = self._solve_limitcycle_bvp()

        if self.BvpOpts['check_stability']:
            assert stability, "Limit cycle is unstable"

        rootflag = self.limitcycle.roots()
        if self.BvpOpts['check_roots']:
            assert rootflag, "Root function failure"

        assert np.linalg.norm(self.limitcycle.dydt(
                 self.limitcycle.y0[:-1])) > 1E-2,\
                "bvp_solution reached steady state (dydt)"

        assert abs(self.limitcycle.y0[-1] - self.NLPdata['TF']
                  )/float(self.NLPdata['TF']) < 1E-1, \
                "bvp_solution has significantly different period (%.3f)"\
                %self.limitcycle.y0[-1]

        assert np.linalg.norm(self.limitcycle.sol[-1] -
                              self.limitcycle.sol[0])\
                < 1E-2, "bvp_solution shows significant error. %.3E"\
                %np.linalg.norm(self.limitcycle.sol[-1] -
                                self.limitcycle.sol[0])

        res = self._match_limitcycle_to_collocation()
        assert res < 1., "bvp_solution significantly different \
        from Collocation solution, error = %.3E" %res


    
    def _create_limitcycle_class(self):
        """
        Creates self.limitcycle object from finished collocation solve
        """

        from CommonFiles.pBase import pBase

        # get guess for [y0,T]
        y0 = self.NLPdata['x_opt'][0].tolist()
        y0 += [self.NLPdata['TF']]

        self.limitcycle = pBase(self.model, self.NLPdata['p_opt'], y0)


    def _solve_limitcycle_bvp(self):
        """
        Calls a few bvp analysis functions
        """
        
        self.limitcycle.solveBVP(method=self.BvpOpts['bvp_method'])
        self.limitcycle.limitCycle()
        return self.limitcycle.check_monodromy()


    def _match_limitcycle_to_collocation(self):
        """
        Offsets the self.limitcycle's solution (t=0 defined as min of
        state 1) to minimize the distance to the collocation solution.
        Returns the residual between the IVP and collocation solutions.
        """

        from scipy.optimize import fmin

        def res(x):
            cost = 0
            for state in xrange(self.NEQ):
                y_bvp = self.limitcycle.lc((self.tgrid +
                            x)%self.limitcycle.y0[-1])[:,state]

                diff = ((y_bvp -
                         np.array(self.NLPdata['x_opt'][:,state])) /
                        max(self.NLPdata['x_opt'][:,state]))

                dist = sum(diff**2)

                cost += dist

            return cost/float(self.NEQ)

        min_t = fmin(res,0,disp=False)

        # Only adjust if solutions are close
        lowest_res = res(min_t)
        if lowest_res < 1: self.limitcycle.offset_t(min_t)
        
        return lowest_res


    def Solve(self, x_init=None, p_init=None):
        """
        Convienence function to set up and solve problem without
        calls to each compartment function. Should be called after
        attaching measurement data.
        """

        self.SetInitialValues(x_init=x_init, p_init=p_init)
        self.ReformModel()
        self.Initialize()
        self.CollocationSetup()
        self.CollocationSolve()
        self.bvp_solution()


        

        

#==========================================================================
# The rest of the class should be primarily concerned with plotting and
# other types of analysis.
#==========================================================================


    def plot_data(self, ax, e, colors=None):
        """
        Plots data points from tdata, ydata for states in e 
        """

        if colors is None: colors = colors_default

        for c,i in enumerate(e):
            # i iterates over states, c over color options
            ax.plot(self.tdata[i], self.ydata[i], 'o',
                    color=colors[c])


    def plot_segments(self, ax, e, v, colors=None):
        """
        Plots line segments specified in v for states in e
        """

        if colors is None: colors = colors_default

        def integrate_segment(x_in, p, t_in):
            """
            Returns ts,xs for integration over one finite element 
            """

            f_p = cs.CVodesIntegrator(self.model)
            f_p.setOption("abstol",1e-8) # tolerance
            f_p.setOption("reltol",1e-8) # tolerance
            f_p.setOption("steps_per_checkpoint",1000)
            f_p.setOption("fsens_err_con",True)
            f_p.setOption("tf",self.NLPdata['TF']/self.NLPdata['NK']) # final time
            f_p.init()
            f_p.setInput(x_in,cs.INTEGRATOR_X0)
            f_p.setInput(p,cs.INTEGRATOR_P)
            f_p.evaluate()
            f_p.reset()
            def out(t):
                f_p.integrate(t)
                return f_p.output().toArray()
            ts = np.linspace(0,self.NLPdata['TF']/self.NLPdata['NK'],self.NLPdata['NK'])
            return (ts+t_in,np.array([out(t) for t in ts]).squeeze())

        p = v[:self.NP].squeeze()
        x_all = v[self.NP:self.NV-self.NMP].reshape(((self.NLPdata['DEG']+1)*self.NLPdata['NK'],self.NVAR))[:,:self.NEQ]
        x = x_all[0::(self.NLPdata['DEG']+1)]
        tg = self.tgrid[0::(self.NLPdata['DEG']+1)]

        for (x,t) in zip(x,tg):
            try:
                ts,xs = integrate_segment(x,p,t)
                for c,i in enumerate(e):
                    ax.plot(ts, self.Amatrix[i].dot(xs.T), '-',
                            color=colors[c])
            except Exception: pass

    def plot_spline(self, ax, e, colors=None):
        """
        Plots the interpolating spline for states in e
        """
        if colors is None: colors = colors_default

        ts = np.linspace(0,self.NLPdata['TF'],200)
        zs = self.sy(ts).T
        for c,i in enumerate(e):
            ax.plot(ts,zs[:,i],':',color=colors[c])

    def plot_collopt(self, ax, e, v=None, colors=None):
        """
        Plot the optimium trajectories obtained from solution of the
        collocation NLP
        """

        if v is None: v = self.NLPdata['v_opt']
        if colors is None: colors = colors_default

        def z(tau, zs):
            """
            Functon to calculate the interpolated values of z^K at a
            given tau (0->1). zs is a matrix with the state values
            within the finite element
            """

            def l(j,t):
                """
                Intermediate values for polynomial interpolation
                """
                tau = self.NLPdata['collocation_points']\
                        [self.NLPdata['cp']][self.NLPdata['DEG']]
                return np.prod(np.array([ 
                        (t - tau[k])/(tau[j] - tau[k]) 
                        for k in xrange(0,self.NLPdata['DEG']+1) if k is not j]))

            
            interp_vector = []
            for i in xrange(self.NEQ):
                interp_vector += [np.sum(np.array([l(j, tau)*zs[j,i]
                                  for j in xrange(0, self.NLPdata['DEG']+1)]))]
            return interp_vector

        x_all = v[self.NP:self.NV-self.NMP].reshape(((self.NLPdata['DEG']+1)*self.NLPdata['NK'],self.NVAR))[:,:self.NEQ]
        xarr = x_all.reshape([self.NLPdata['NK'],self.NLPdata['DEG']+1,self.NEQ])

        ts = np.linspace(0,1)
        t_out = []
        x_out = []
        for i in xrange(self.NLPdata['NK']):
            t_out += [ts*self.h + i*self.h]
            x_out += [np.array([z(t, xarr[i]) for t in ts])]


        for c,i in enumerate(e):
            for j in xrange(self.NLPdata['NK']):
                ax.plot(t_out[j], self.Amatrix[i].dot(x_out[j].T), '-',
                        color=colors[c])
            ax.plot(self.tgrid,
                    self.Amatrix[i].dot(xarr.reshape([self.NLPdata['NK']*(self.NLPdata['DEG']+1),
                                                      self.NEQ]).T), '.',
                    color=colors[c])


    def plot_opt(self, ax, e, colors=None):
        """
        Plot the limit cycle of the optimal trajectory
        """

        if colors is None: colors = colors_default

        if not hasattr(self, 'limitcycle'):
            self.bvp_solution()

        for c,i in enumerate(e):
            ax.plot(self.limitcycle.ts,
                    self.Amatrix[i].dot(self.limitcycle.sol.T),
                    color=colors[c])

    def plot_format_circadian(self, ax):
        """
        Takes an input axes (polar projection) and formats it to
        have 0->TF north-polar axes. Re-names period to be 24 hours, be
        careful if this is not the case
        """

        import matplotlib.pylab as plt
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_xticks(np.linspace(0,2*np.pi,8, endpoint=False).tolist())
        ax.set_xticklabels(('0','3','6','9','12','15','18 ','21'))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        plt.setp(ax.get_yticklabels(), color='0.4')

        for line in ax.lines:
            line.set_xdata(line.get_xdata()/self.NLPdata['TF']*2*np.pi)


    def CollocationPlot(self, init=True, opt='bvp', spline=True,
                        show=True, e=None):
        """
        Plots the initial and final trajectories of the collocation
        optimization
        
        init: (bool) Plot NLPdata['v_init']
        opt: 'col', 'bvp', or None
        returns: None
        """
        import matplotlib.pylab as plt

        if not e:
            if self.NM > 3: e = range(4)
            else: e = range(self.NM)

        numfigs = init + bool(opt) + spline
        height = 2
        width = 2*numfigs

        fig = plt.figure(figsize=(width, height))
        ax = [[]]*numfigs
        for i in xrange(numfigs):
            ax[i] = fig.add_subplot(1, numfigs, i)

        p = numfigs-1

        if init:
            self.plot_data(ax[p], e)
            self.plot_segments(ax[p], e, self.NLPdata['v_init'])
            ax[p].set_title("Initial")
            p -= 1

        if spline:
            self.plot_data(ax[p], e)
            self.plot_spline(ax[p], e)
            ax[p].set_title("Spline")
            p -= 1

        if opt:
            self.plot_data(ax[p], e)
            # if opt is 'opt': self.plot_opt(ax[p], e)
            if opt is 'col': self.plot_collopt(ax[p], e, self.NLPdata['v_opt'])
            else: self.plot_segments(ax[p], e, self.NLPdata['v_opt'])
            ax[p].set_title("Optimal")
            p -= 1

        # for axis in ax:
        #     self.plot_format_circadian(axis)

        if show: plt.show()


if __name__ == "__main__":


    from CommonFiles.pBase import pBase
    from CommonFiles.tyson2statemodel import model,paramset,y0in

    new = LimitCycleCollocation(model(), paramset)
    new.ReformModel()
    new.CoefficentSetup()
    new.Initialize()
    new.CollocationSetup()
    new.CollocationSolve()
