# COMMON IMPORTS
from itertools import izip
import numpy  as np
from matplotlib.ticker import MaxNLocator

# THIRD PARTY IMPORTS
import casadi as cs

# MY IMPORTS
from Utilities import fnlist, get_ind, spline
from Utilities import colors as colors_default



class Collocation:
    """
    Class to solve for a parameterization of a given model which
    minimizes the distance between x(t) and given data points. 
    """

    def __init__(self, model=None):

        # Define some useful constants
        self.NICP = 1    # Number of collocation points per interval?
        self.NK   = 10   # Shooting discretization
        self.TF   = 24.0 # End time
        self.DEG  = 5    # Degree of interpolating polynomelf.al

        # Set up some default placeholders for bounds
        self.PARMAX = 1E+2
        self.PARMIN = 1E-3
        self.XMAX   = 1E+2
        self.XMIN   = 1E-3


        # Storage Dictionary for solver parameters.
        # These options are my own, used at various locations throughout
        # the class
        self.NLPdata = {
            'FPTOL'             : 1E-3,
            'ObjMethod'         : 'lsq', # lsq or laplace
            'FPgaurd'           : False,
            'CONtol'            : 0,
            'f_minimize_weight' : 1,
            'stability'         : False # False or numerical weight
        }

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

        # Boundary Value Problem Options. Here we can change integration
        # toleraces, method options, etc.
        self.BvpOpts = {
            'Y0TOL'           : 1E-3,
            'bvp_method'      : 'periodic',
            'findstationary'  : False,
            'check_stability' : True,
            'check_roots' : True
        }


        self.minimize_f = None

        if model:
            self.AttachModel(model)

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
        self.h = self.TF/self.NK/self.NICP

        # Coefficients of the collocation equation
        C = np.zeros((self.DEG+1,self.DEG+1))
        # Coefficients of the continuity equation
        D = np.zeros(self.DEG+1)
        # Coefficients for integration
        E = np.zeros(self.DEG+1)

        # Collocation point
        tau = cs.ssym("tau")
          
        # All collocation time points
        tau_root = collocation_points[cp][self.DEG]
        T = np.zeros((self.NK,self.DEG+1))
        for i in range(self.NK):
          for j in range(self.DEG+1):
                T[i][j] = self.h*(i + tau_root[j])

        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at
        # the collocation point
        for j in range(self.DEG+1):
            L = 1
            for j2 in range(self.DEG+1):
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
            for j2 in range(self.DEG+1):
                lfcn.setInput(tau_root[j2])
                lfcn.setFwdSeed(1.0)
                lfcn.evaluate(1,0)
                C[j][j2] = lfcn.fwdSens()

            # lint = cs.CVodesIntegrator(lfcn)

            tg = np.array(tau_root)*self.h
            for k in range(self.NK*self.NICP):
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

        # Set up PARMAX and PARMIN variables. Check if they are a
        # vector, if not resize to number of parameters.
        try:
            assert len(self.PARMAX) == len(self.PARMIN) == self.NP, \
            "Parameter bounds not correct length"
        except TypeError:
            self.PARMAX = [self.PARMAX] * self.NP
            self.PARMIN = [self.PARMIN] * self.NP
            
        # Set up XMAX and XMIN variables. Check if they are a
        # vector, if not resize to number of state variables
        try:
            assert len(self.XMAX) == len(self.XMIN) == self.NEQ, \
            "State variable bounds not correct length"
        except TypeError:
            self.XMAX = [self.XMAX] * self.NEQ
            self.XMIN = [self.XMIN] * self.NEQ


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


    def ReformModel(self, sensids=[]):
        """
        Reform self.model to conform to the standard (implicit) form
        used by the rest of the collocation calculations. Should also
        append sensitivity states (dy/dp_i) given as a list to the end
        of the ode model.

        Monodromy - calculate a symbolic monodromy matrix by integrating
        dy/dy_0,i 
        """
        
        # Find total number of variables for symbolic allocation
        self.monodromy = bool(self.NLPdata['stability'])
        self.NSENS = len(sensids) + self.NEQ*self.monodromy
        nsvar = self.NSENS * self.NEQ
        self.NVAR = (1 + self.NSENS)*self.NEQ

        # Allocate symbolic vectors for the model
        t     = self.model.inputSX(cs.DAE_T)    # time
        u     = cs.ssym("u", 0, 1)              # control (empty)
        xd_in = self.model.inputSX(cs.DAE_X)    # differential state
        s     = cs.ssym("s", nsvar, 1)          # sensitivities
        xa    = cs.ssym("xa", 0, 1)             # algebraic state (empty)
        xddot = cs.ssym("xd", self.NEQ + nsvar) # differential state dt
        p     = self.model.inputSX(2)           # parameters

        # Symbolic function (from input model)
        ode_rhs = self.model.outputSX()

        # symbolic jacobians
        jac_x = self.model.jac(cs.DAE_X, cs.DAE_X)   
        jac_p = self.model.jac(cs.DAE_P, cs.DAE_X)


        sens_rhs = []
        for index, state in enumerate(sensids):
            s_i = s[index*self.NEQ:(index + 1)*self.NEQ]
            rhs_i = jac_x.mul(s_i) + jac_p[:, state]
            sens_rhs += [rhs_i]

        offset = len(sensids)*self.NEQ
        if self.monodromy:
            for i in xrange(self.NEQ):
                s_i = s[offset + i*self.NEQ:offset + (i+1)*self.NEQ]
                rhs_i = jac_x.mul(s_i)
                sens_rhs += [rhs_i]
            
        sens_rhs = cs.vertcat(sens_rhs).reshape((nsvar,1))
            

        ode = xddot[:self.NEQ] - ode_rhs
        sens = xddot[self.NEQ:] - sens_rhs
        xd = cs.vertcat([xd_in, s])
        tot = cs.vertcat([ode, sens])
        
        self.rfmod = cs.SXFunction([t,xddot,xd,xa,u,p], [tot])

        self.rfmod.init()



    def MeasurementMatrix(self, matrix, measurement_pars=[]):
        """
        Attaches the matrix A which specifies the measured output states. 
        y(t) = A x(t), where x are the internal states.
        """
        
        height, width = matrix.shape
        assert width == self.NEQ, "Measurement Matrix shape mis-match"
        self.NM = height
        self.Amatrix = matrix

        self.ydata = self.NM*[[]]
        self.tdata = self.NM*[[]]
        self.edata = self.NM*[[]]

        if not measurement_pars: measurement_pars = [False]*self.NM
        else: assert len(measurement_pars) == self.NM, \
                'Measurement Pars shape mis-match'
        
        self.mpars = measurement_pars
        self.NMP = sum(self.mpars)

        self.YMAX = matrix.dot(np.array(self.XMAX))
        self.YMIN = matrix.dot(np.array(self.XMIN))



    def AttachMeasurementData(self, measurement, times, values,
                              errors=False):
        """
        Attach each measurements's data points
        errors should be standard deviations in the measurements.

        Requirements: AttachModel
        """
        
        # Recast Inputs
        values = np.array(values)
        times = np.array(times)
        
        # If error not provided, set to ones)
        if errors is False: errors = np.ones(len(times))
        else: errors = np.array(errors)

        # Check appropriate sizes
        assert(len(times) == len(values) == len(errors))

        # Make sure the measurement index is appropriate
        assert(self.NM > measurement >= 0)
        #
        # Make sure data is within t-bounds. (this might want to throw
        # an error as well)
        if np.any(times > self.TF):
            print "BOUNDS WARNING: measurement in measurement %s" \
                    %measurement,
            print " has time above TF" 

        if np.any(values > self.YMAX[measurement]):
            print "BOUNDS WARNING: measurement in measurement %s" \
                    %measurement,
            print " has value above YMAX" 
            invalid_ind = np.where(np.array(values) >
                                   self.YMAX[measurement])[0]
            values[invalid_ind] = self.YMAX[measurement]

        if np.any(values < self.YMIN[measurement]):
            print "BOUNDS WARNING: measurement in measurement %s" \
                    %measurement,
            print " has value below YMIN" 
            invalid_ind = np.where(np.array(values) <
                                   self.YMIN[measurement])[0]
            values[invalid_ind] = self.YMIN[measurement]

        # make sure data is in acending order.
        times = times%self.TF
        sort_inds = times.argsort()

        values = values[sort_inds].tolist()
        times  = times[sort_inds].tolist()
        errors = errors[sort_inds].tolist()

        self.ydata[measurement] = values
        self.tdata[measurement] = times
        self.edata[measurement] = errors


    
    def AttachMinimizeFunction(self, f):
        """
        Attach a function that will specify an objective function to
        minimize (negative for maximize) over the entire domain,
        (minimize integral of f(x, p) over [0,T]) in addition to fitting
        model trajectories
        """

        self.minimize_f = f



    def SetInitialValues(self, x_init=None, p_init=None):
        """
        This function will set the intial conditions for the NLP
        problem, x_init and p_init. Initial guesses, if provided, will
        be stored appropriately. If not provided, they will be estimated
        using the Trajectory and Parameter Estimation routines.

        Requirements: AttachModel, AttachMeasurementFuction
        """

        # We need an interpolating function function for the
        # measurements, we'll place it here.
        sfactor = 5E-3
        self.sy = fnlist([])
        for t, y in izip(self.tdata, self.ydata):
            # Duplicate the first entry.
            tvals = list(t)
            tvals.append(t[0]+self.TF)
            yvals = list(y)
            yvals.append(y[0])
            self.sy += [spline(tvals, yvals, sfactor)]


        # Interpolate from measurement data
        if x_init is None: x_init = self.TrajectoryEstimation()
        else:
            assert x_init.shape == (len(self.tgrid), self.NEQ), \
                "Shape Mismatch"
            assert (np.all(x_init < self.XMAX) &
                    np.all(x_init > self.XMIN)), "Bounds Error"

        self.x_init = x_init

        # Interpolate from slopes of measurement data
        if p_init is not None: self.p_init = p_init
        else:
            self.p_init = self.ParameterEstimation()


    def TrajectoryEstimation(self):
        """
        This function will be responsible for generating appopriate
        guess values for x_init from the measurement variables y
        """

        from scipy.optimize import minimize
        node_ts = self.tgrid.reshape((self.NK, self.DEG+1))[:,0]

        # Loop through start of each finite element (node_ts), solve for
        # optimum state variables from the state constructure
        # measurement functions.
        xopt = []

        bounds = [[xmin, xmax] for xmin, xmax in
                  zip(self.XMIN,self.XMAX)]

        options = {'disp' : False}

        for t in node_ts:
            # Initial guess for optimization. If its the first point,
            # start with [1]*NEQ, otherwise use the result from the
            # previous finite element.
            
            if xopt == []: iguess = np.ones(self.NEQ)
            else: iguess = xopt[-1]

            y = self.sy(t)

            # Inefficient but simple - might need to rethink.
            # Distance weight: to improve identifiability issues, return
            # the solution closest to the previous iterate (or [1]'s)
            dw = 1E-5
            def min_func(x):
                dist = np.linalg.norm(iguess - x)
                ret = sum((self.Amatrix.dot(x) - y)**2) + dw * dist
                return ret
            xopt += [minimize(min_func, iguess, bounds=bounds,
                              method='L-BFGS-B', options=options)['x']]

        x_init_course = np.array(xopt)

        # Resample x_init to get all the needed points
        self.sx = fnlist([])

        def flat_factory(x):
            """ Protects against nan's resulting from flat trajectories
            in spline curve """
            return lambda t, d: (np.array([x] * len(t)) if d is 0 else
                                 np.array([0] * len(t)))

        for x in np.array(x_init_course).T:
            tvals = list(node_ts)
            tvals.append(tvals[0] + self.TF)
            xvals = list(x)
            xvals.append(x[0])
            
            if np.linalg.norm(xvals - xvals[0]) < 1E-8:
                # flat profile
                self.sx += [flat_factory(xvals[0])]

            else: self.sx += [spline(tvals, xvals, 0)]

        x_init = self.sx(self.tgrid,0).T

        below = np.where(x_init < self.XMIN)
        above = np.where(x_init > self.XMAX)

        for t_ind, state in zip(*below):
            x_init[t_ind, state] = self.XMIN[state]

        for t_ind, state in zip(*above):
            x_init[t_ind, state] = self.XMAX[state]

        return x_init



    def ParameterEstimation(self):
        """
        Here we will set up an NLP to estimate the initial parameter
        guess (potentially along with unmeasured state variables) by
        minimizing the difference between the calculated slope at each
        shooting node (a function of the parameters) and the measured
        slope (using a spline interpolant)
        """

        # Here we must interpolate the x_init data using a spline. We
        # will differentiate this spline to get intial values for
        # the parameters
        
        node_ts = self.tgrid.reshape((self.NK, self.DEG+1))[:,0]
        try:
            f_init = self.sx(self.tgrid,1).T
        except AttributeError:
            # Create interpolation object
            sx = fnlist([])
            def flat_factory(x):
                """ Protects against nan's resulting from flat trajectories
                in spline curve """
                return lambda t, d: (np.array([x] * len(t)) if d is 0 else
                                     np.array([0] * len(t)))
            for x in np.array(self.x_init).T:
                tvals = list(node_ts)
                tvals.append(tvals[0] + self.TF)
                xvals = list(x.reshape((self.NK, self.DEG+1))[:,0])
                xvals.append(x[0])
                
                if np.linalg.norm(xvals - xvals[0]) < 1E-8:
                    # flat profile
                    sx += [flat_factory(xvals[0])]

                else: sx += [spline(tvals, xvals, 0)]

            f_init = sx(self.tgrid,1).T

        # set initial guess for parameters
        logmax = np.log10(np.array(self.PARMAX))
        logmin = np.log10(np.array(self.PARMIN))
        p_init = 10**(logmax - (logmax - logmin)/2)

        f = self.model
        V = cs.MX("V", self.NP)
        par = V

        # Allocate the initial conditions
        pvars_init = np.ones(self.NP)
        pvars_lb = np.array(self.PARMIN)
        pvars_ub = np.array(self.PARMAX)
        pvars_init = p_init

        xin = []
        fin = []

        # For each state, add the (variable or constant) MX
        # representation of the measured x and f value to the input
        # variable list.
        for state in xrange(self.NEQ):
            xin += [cs.MX(self.x_init[:,state])]
            fin += [cs.MX(f_init[:,state])]

        xin = cs.horzcat(xin)
        fin = cs.horzcat(fin)

        # For all nodes
        res = []
        xmax = self.x_init.max(0)
        for ii in xrange((self.DEG+1)*self.NK):
            f_out = f.call(cs.daeIn(t=self.tgrid[ii],
                                    x=xin[ii,:].T, p=par))[0]
            res += [cs.sumAll(((f_out - fin[ii,:].T) / xmax)**2)]

        F = cs.MXFunction([V],[cs.sumAll(cs.vertcat(res))])
        F.init()
        parsolver = cs.IpoptSolver(F)

        for opt,val in self.IpoptOpts.iteritems():
            if not opt == "expand_g":
                parsolver.setOption(opt,val)

        parsolver.init()
        parsolver.setInput(pvars_init,cs.NLP_X_INIT)
        parsolver.setInput(pvars_lb,cs.NLP_LBX)
        parsolver.setInput(pvars_ub,cs.NLP_UBX)
        parsolver.solve()
        
        success = parsolver.getStat('return_status') == 'Solve_Succeeded'
        assert success, "Parameter Estimation Failed"

        self.pcost = float(parsolver.output(cs.NLP_COST))

        
        pvars_opt = np.array(parsolver.output( \
                            cs.NLP_X_OPT)).flatten()

        if success: return pvars_opt
        else: return False


    def Initialize(self):
        """
        Uses the finished p_init and x_init values to finalize the
        NLPd structure with full matricies on state bounds, parameter
        bounds, etc.

        Requirements: AttachModel, AttachData, SetInitialValues,
        (ParameterEstimation optional)
        """

        NLPd = self.NLPdata
        nsvar = self.NEQ*self.NSENS
        
        nsy = self.NEQ**2 if self.monodromy else 0
        nsp = nsvar - nsy
            

        p_init = self.p_init
        x_init = self.x_init
        xD_init = np.zeros((len(self.tgrid), self.NEQ + nsvar))
        xD_init[:,:self.NEQ] = x_init

        # Set dy_i/dy_0,j = 1 if i=j
        ics = np.eye(self.NEQ).flatten()
        ics = ics[np.newaxis,:].repeat(len(self.tgrid),axis=0)

        if self.monodromy:
            xD_init[:,-nsy:] = ics
            iclist = np.eye(self.NEQ).flatten().tolist()
        else:
            iclist = []

        
        if type(self.XMAX) is np.ndarray: self.XMAX = self.XMAX.tolist()
        if type(self.XMIN) is np.ndarray: self.XMIN = self.XMIN.tolist()

        # Algebraic state bounds and initial guess
        NLPd['xA_min']  = np.array([])
        NLPd['xA_max']  = np.array([])
        NLPd['xAi_min'] = np.array([])
        NLPd['xAi_max'] = np.array([])
        NLPd['xAf_min'] = np.array([])
        NLPd['xAf_max'] = np.array([])
        NLPd['xA_init'] = np.array((self.NK*self.NICP*(self.DEG+1))*[[]])

        # Control bounds
        NLPd['u_min']  = np.array([])
        NLPd['u_max']  = np.array([])
        NLPd['u_max']  = np.array([])
        NLPd['u_init'] = np.array((self.NK*self.NICP*(self.DEG+1))*[[]])
        
        # Differential state bounds and initial guess
        NLPd['xD_min']  =  np.array(self.XMIN + [-1E6]*nsvar)
        NLPd['xD_max']  =  np.array(self.XMAX + [+1E6]*nsvar)
        NLPd['xDi_min'] =  np.array(self.XMIN + [0]*nsp +
                                    iclist)
        NLPd['xDi_max'] =  np.array(self.XMAX + [0]*nsp +
                                    iclist)
        NLPd['xDf_min'] =  np.array(self.XMIN + [-1E6]*nsvar)
        NLPd['xDf_max'] =  np.array(self.XMAX + [+1E6]*nsvar)
        NLPd['xD_init'] =  xD_init
        # needs to be specified for every time interval
        

        # Parameter bounds and initial guess
        NLPd['p_min']  = np.array(self.PARMIN)
        NLPd['p_max']  = np.array(self.PARMAX)
        NLPd['p_init'] = p_init



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

        # Initial conditions for measurement adjustment
        MP = V[self.NV-self.NMP:]
        vars_init[self.NV-self.NMP:] = np.ones(self.NMP)
        vars_lb[self.NV-self.NMP:] = 0.1*np.ones(self.NMP) 
        vars_ub[self.NV-self.NMP:] = 10*np.ones(self.NMP)



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


        # Get Linear Interpolant for YDATA from TDATA
        objlist = []
        # xarr = np.array([V[self.NP:][i] for i in \
        #         xrange(self.NEQ*self.NK*(self.DEG+1))])
        # xarr = xarr.reshape([self.NK,self.DEG+1,self.NEQ])
        
        # List of the times when each finite element starts
        felist = self.tgrid.reshape((self.NK,self.DEG+1))[:,0]
        felist = np.hstack([felist, self.tgrid[-1]])

        def z(tau, zs):
            """
            Functon to calculate the interpolated values of z^K at a
            given tau (0->1). zs is a matrix with the symbolic state
            values within the finite element
            """

            def l(j,t):
                """
                Intermediate values for polynomial interpolation
                """
                tau = self.NLPdata['collocation_points']\
                        [self.NLPdata['cp']][self.DEG]
                return np.prod(np.array([ 
                        (t - tau[k])/(tau[j] - tau[k]) 
                        for k in xrange(0,self.DEG+1) if k is not j]))

            
            interp_vector = []
            for i in xrange(self.NEQ): # only state variables
                interp_vector += [np.sum(np.array([l(j, tau)*zs[j][i]
                                  for j in xrange(0, self.DEG+1)]))]
            return interp_vector

        # Set up Objective Function by minimizing distance from
        # Collocation solution to Measurement Data

        # Number of measurement functions
        for i in xrange(self.NM):

            # Number of sampling points per measurement]
            for j in xrange(len(self.tdata[i])):

                # the interpolating polynomial wants a tau value,
                # where 0 < tau < 1, distance along current element.
                
                # Get the index for which finite element of the tdata 
                # values
                feind = get_ind(self.tdata[i][j],felist)[0]

                # Get the starting time and tau (0->1) for the tdata
                taustart = felist[feind]
                tau = (self.tdata[i][j] - taustart)*(self.NK+1)/self.TF

                x_interp = z(tau, XD[feind][0])
                # Broken in newest numpy version, likely need to redo
                # this whole file with most recent versions
                y_model = self.Amatrix[i].dot(x_interp)

                # Add measurement scaling
                if self.mpars[i]: y_model *= MP[sum(self.mpars[:i])]

                # Using relative diff's is probably messing up weights
                # in Identifiability
                diff = (y_model - self.ydata[i][j])

                if   self.NLPdata['ObjMethod'] == 'lsq':
                    dist = (diff**2/self.edata[i][j]**2)

                elif self.NLPdata['ObjMethod'] == 'laplace':
                    dist = cs.fabs(diff)/np.sqrt(self.edata[i][j])
                
                try: dist *= self.NLPdata['state_weights'][i]
                except KeyError: pass
                    
                objlist += [dist]

        # Minimization Objective
        if self.minimize_f:
            # Function integral
            f_integral = 0
            # For each finite element
            for i in xrange(self.NK):
                # For each collocation point
                fvals = []
                for j in xrange(self.DEG+1):
                    fvals += [self.minimize_f(XD[i][0][j], P)]
                # integrate with weights
                f_integral += sum([fvals[k] * self.E[k] for k in
                                   xrange(self.DEG+1)])

            objlist += [self.NLPdata['f_minimize_weight']*f_integral]



        # Stability Objective (Floquet Multipliers)
        if self.monodromy:
            s_final = XD[-1,-1,-1][-self.NEQ**2:]
            s_final = s_final.reshape((self.NEQ,self.NEQ))
            trace = sum([s_final[i,i] for i in xrange(self.NEQ)])
            objlist += [self.NLPdata['stability']*(trace - 1)**2]




        # Objective function of the NLP
        obj = cs.sumAll(cs.vertcat(objlist))
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
                      self.NMP].reshape(((self.DEG+1)*self.NK,
                                         self.NVAR))
        self.NLPdata['x_opt'] = x_opt[:,:self.NEQ]
        self.NLPdata['s_opt'] = \
        x_opt[:,self.NEQ:].reshape((len(self.tgrid), self.NSENS,
                                    self.NEQ))
        self.NLPdata['xd_opt'] = x_opt[1::(self.DEG+1)]
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

        assert abs(self.limitcycle.y0[-1] - self.TF
                  )/float(self.TF) < 1E-1, \
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
        y0 += [self.TF]

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
            f_p.setOption("tf",self.TF/self.NK) # final time

            ts = np.linspace(0,self.TF/self.NK,self.NK)
            f_p_sim = cs.Simulator(f_p, ts)
            f_p_sim.init()
            f_p_sim.setInput(x_in,cs.INTEGRATOR_X0)
            f_p_sim.setInput(p,cs.INTEGRATOR_P)
            f_p_sim.evaluate()

            return (ts+t_in, f_p_sim.output().toArray())

        p = v[:self.NP].squeeze()
        x_all = v[self.NP:self.NV - 
                  self.NMP].reshape(((self.DEG+1) * self.NK,
                                     self.NVAR))[:,:self.NEQ]
        x = x_all[0::(self.DEG+1)]
        tg = self.tgrid[0::(self.DEG+1)]

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

        ts = np.linspace(0,self.TF,200)
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
                        [self.NLPdata['cp']][self.DEG]
                return np.prod(np.array([ 
                        (t - tau[k])/(tau[j] - tau[k]) 
                        for k in xrange(0,self.DEG+1) if k is not j]))

            
            interp_vector = []
            for i in xrange(self.NEQ):
                interp_vector += [np.sum(np.array([l(j, tau)*zs[j,i]
                                  for j in xrange(0, self.DEG+1)]))]
            return interp_vector

        x_all = v[self.NP:self.NV-self.NMP].reshape(((self.DEG+1)*self.NK,self.NVAR))[:,:self.NEQ]
        xarr = x_all.reshape([self.NK,self.DEG+1,self.NEQ])

        ts = np.linspace(0,1)
        t_out = []
        x_out = []
        for i in xrange(self.NK):
            t_out += [ts*self.h + i*self.h]
            x_out += [np.array([z(t, xarr[i]) for t in ts])]


        for c,i in enumerate(e):
            for j in xrange(self.NK):
                ax.plot(t_out[j], self.Amatrix[i].dot(x_out[j].T), '-',
                        color=colors[c])
            ax.plot(self.tgrid,
                    self.Amatrix[i].dot(xarr.reshape([self.NK*(self.DEG+1),
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
            line.set_xdata(line.get_xdata()/self.TF*2*np.pi)


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
    # from CommonFiles.degmodelFinal import model, paramset, y0in
    from CommonFiles.Models.tyson2statemodel import model,paramset,y0in
    # from CommonFiles.leloup16model import model,paramset,NEQ,NP,y0in
    #from CommonFiles.HenrysModel import model,paramset,NEQ,NP,y0in

    nk = 20

    np.random.seed(1000)

    orig = pBase(model(), paramset, y0in)
    orig.limitCycle()

    ts = np.linspace(0, y0in[-1], nk, endpoint=False)
    sol = orig.lc(ts)

    #errors = 2*np.random.rand(*sol.shape)
    errors = np.ones(sol.shape)


    # Generate New Model missing first state
    # modl = model()
    # ins = modl.inputsSX()
    # out = modl.outputSX()

    # newmod = cs.SXFunction([ins[0],ins[1][:-1],ins[2][:-1],ins[3]],[out[:-1]])



    test = Collocation(model()); print "Set Up Completed"

    test.TF = y0in[-1]
    test.NK = 20
    test.NLPdata['ObjMethod'] = 'lsq'
    test.NLPdata['print_level'] = 5
    test.NLPdata['f_minimize_weight'] = 10
    test.IpoptOpts['max_iter'] = 2000
    test.IpoptOpts['max_cpu_time'] = 60*20
    # test.IpoptOpts['linear_solver'] = 'mumps'
    test.IpoptOpts['tol'] = 1E-8
    test.NLPdata['FPgaurd'] = False

    test.PARMAX = 1E+1
    test.PARMIN = 1E-3
    test.XMAX = 1E+1
    test.XMIN = 1E-4

    test.CoefficentSetup(); print "Coefficient Setup"

    A = np.eye(test.NEQ)
    test.MeasurementMatrix(A)
    
    y_exact = np.array([A.dot(sol[i]) for i in xrange(len(sol))]) 
    y_error = (y_exact*(1 + 0.15*(np.random.rand(*y_exact.shape) - 0.5))
               + 0.05*y_exact.max()*np.random.rand(*y_exact.shape))
    errors = (y_exact*0.05 + 0.01*y_exact.max())


    for i in xrange(test.NM): test.AttachMeasurementData(i, ts, y_error[:,i], errors[:,i])
    print "Data Attached"

    f_minimize = lambda x,p: x[0]*p[3] + x[1]*p[5]
    # test.AttachMinimizeFunction(f_minimize)

    # test.SetInitialValues(p_init=orig.paramset, x_init=orig.lc(test.tgrid))
    # test.SetInitialValues()
    # # test.SetInitialValues(x_init = orig.lc(test.tgrid))
    # print "Inital Values Completed"


    # test.ReformModel()
    # test.Initialize(); print "Initialization Completed"
    # test.CollocationSetup()
    # test.CollocationSolve()
    # try: test.bvp_solution()
    # except Exception as ex:
    #     test.limitcycle.limitCycle()
    #     print ex
    #     print test.limitcycle.floquet_multipliers

    test.Solve()

    test.CollocationPlot(opt='col', show=False); print "Collocation Plot"

    import matplotlib.pyplot as plt
    plt.figure()

    for i in xrange(min(test.NEQ,4)):
        plt.plot(test.limitcycle.ts, test.limitcycle.sol[:,i], ':',
                 color=colors_default[i])
        plt.plot(ts, sol[:,i], 'o', color = colors_default[i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test.limitcycle.ts, test.limitcycle.sol)
    test.plot_collopt(ax, [0,1])
    # plt.plot(test.limitcycle.ts, [f_minimize(y, test.limitcycle.paramset) for y in test.limitcycle.sol])

    plt.show()

    # plt.figure(2)
    # for i in xrange(len(test.flist)):
    #     plt.plot(test.limitcycle.ts, test.flist(test.limitcycle.sol.T)[i], color=test.colors[i])
    #     plt.plot(test.tdata[i], test.zdata[i], '.', color=test.colors[i])
    # 
    # plt.show()



