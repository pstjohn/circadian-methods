import numpy  as np
import scipy  as sc
import casadi as cs

class ForcedOscil(object):
    """ Class to handle the intergration and evaluation of models with a
    forced trajectory """

    def __init__(self, base_class, res=20, tf=None):
        """ Attach an existing limit cycle class and define control
        resolution """

        # Attach base class
        self.base_class = base_class
        self.nk = res

        # Set up end time. If empty, use limitcycle period
        if tf is None: self.tf = base_class.y0[-1]
        else: self.tf = tf
        
        # Elevate attributes from the base class
        self.NEQ      = self.base_class.NEQ
        self.NP       = self.base_class.NP
        self.ydict    = self.base_class.ydict
        self.pdict    = self.base_class.pdict
        self.model    = self.base_class.model
        self.paramset = self.base_class.paramset
        self.y0       = self.base_class.y0[:-1]

        self.force_param = {}

        self.coll_setup_dict = setup_collocation(self.tf, self.nk,
                                                 deg=5)
        self.tgrid   = self.coll_setup_dict['tgrid']
        self.tgrid_u = self.coll_setup_dict['tgrid_u']
        self.h       = self.coll_setup_dict['h']
        self.deg     = self.coll_setup_dict['deg']
        self.coll_setup_dict['xD_min'] = np.array([0]*self.NEQ)
        self.coll_setup_dict['xD_max'] = np.array([1E+3]*self.NEQ)

        # Options dictionary
        self.int_opt = {
            'int_abstol'        : 1E-6,
            'int_reltol'        : 1E-6,
            'bvp_tol'           : 1E-6,
            'bvp_linear_solver' : 'ma57',
            'bvp_constr_tol'    : 1E-4,
            'bvp_print_level'   : 0,
        }

        self._reform_model()
        self.need_bvp_setup = True


    def _reform_model(self):
        """
        Reform self.model to conform to the standard (implicit) form
        used by the rest of the collocation calculations.
        """
        #
        # Allocate symbolic vectors for the model
        t     = self.model.inputSX(cs.DAE_T)    # time
        xd    = self.model.inputSX(cs.DAE_X)    # differential state
        xddot = cs.ssym("xd", self.NEQ)         # differential state dt
        p     = self.model.inputSX(2)           # parameters

        # Symbolic function (from input model)
        ode_rhs = self.model.outputSX()
        ode = xddot[:self.NEQ] - ode_rhs
        self.rfmod = cs.SXFunction([t,xddot,xd,p], [ode])
        self.rfmod.init()

    def ForcingParam(self, par_label, func, rel=True):
        """ Add to forcing parameter dictionary.
        if rel: y_vals = initial_param * func(tgrid_u)
        else: y_vals = func(tgrid_u) """
        
        if rel: yvals = (self.paramset[self.pdict[par_label]] * 
                         np.array([func(t) for t in self.tgrid_u]))
        else: yvals = np.array([func(t) for t in self.tgrid_u])
        self.force_param[par_label] = yvals

    def _get_paramset(self, t):
        param = np.array(self.paramset)
        segment = int(t/self.h)%self.nk
        for par, yvals in self.force_param.iteritems():
            param[self.pdict[par]] = yvals[segment]
        return param

    def Integrate(self, t, sim=True, res=500, pars=None):
        """ Integrate the forced oscillator from t = 0 to t. sim=True,
        return vector of values. """ 

        # Get parameters over the limit cycle
        if pars is None:
            pars = np.array([self._get_paramset(tj) for tj in
                             self.tgrid_u]).flatten()

        apars = pars.reshape((self.nk, self.NP))
        y = self.y0

        integrator = cs.CVodesIntegrator(self.model)
        integrator.setOption('abstol', self.int_opt['int_abstol'])
        integrator.setOption('reltol', self.int_opt['int_reltol'])
        integrator.setOption('tf', self.h)
        integrator.init()

        t_i = 0

        if not sim:

            while t_i < t - self.h + 1E-5:
                element_index = int((t_i/self.h)%self.nk)
                integrator.setInput(y, cs.INTEGRATOR_X0)
                integrator.setInput(apars[element_index], cs.INTEGRATOR_P)
                integrator.evaluate()
                y = integrator.output()
                t_i += self.h

            if t - t_i > 1E-5:
                element_index = int((t_i/self.h)%self.nk)
                integrator.setOption('tf', t-t_i)
                integrator.init()
                integrator.setInput(y, cs.INTEGRATOR_X0)
                integrator.setInput(apars[element_index], cs.INTEGRATOR_P)
                integrator.evaluate()
                y = integrator.output()

            return t, y

        else:
            # Allow for specific ts requests in res
            try: ts = np.linspace(0, t, res)
            except TypeError: ts = res
            sol = np.zeros((0, self.NEQ))

            while t_i < t - self.h + 1E-5:
                element_index = int((t_i/self.h)%self.nk)
                ts_i = ts[np.all([ts < t_i + self.h, ts >= t_i], 0)] - t_i
                sim_i = cs.Simulator(integrator,
                                     np.hstack([0, ts_i, self.h]))
                sim_i.init()
                sim_i.setInput(y, cs.INTEGRATOR_X0)
                sim_i.setInput(apars[element_index], cs.INTEGRATOR_P)
                sim_i.evaluate()
                out = sim_i.output().toArray()
                sol = np.vstack([sol, out[1:-1]])
                y = out[-1]
                t_i += self.h

            if t - t_i > 1E-5:
                element_index = int((t_i/self.h)%self.nk)
                ts_i = ts[np.all([ts < t_i + self.h, ts >= t_i], 0)] - t_i
                sim_i = cs.Simulator(integrator,
                                     np.hstack([0, ts_i, t-t_i]))
                sim_i.init()
                sim_i.setInput(y, cs.INTEGRATOR_X0)
                sim_i.setInput(apars[element_index], cs.INTEGRATOR_P)
                sim_i.evaluate()
                out = sim_i.output().toArray()
                sol = np.vstack([sol, out[1:-1]])
                y = out[-1]

            # Close enough.
            elif len(sol) == len(ts) - 1:
                sol = np.vstack([sol, y])

            return ts, sol

    def _calc_initial_guess(self, periods=4, pars=None):
        """ function to calculate an initial guess trajectory for the
        bvp_solution method """

        t, y = self.Integrate((periods - 1)*self.tf, sim=False)
        self.y0 = y
        ts, ys = self.Integrate(self.tf, sim=True, res=self.tgrid, pars=pars)
        return ts, ys

    def _interpolate_bvp_solution(self):
        """ Create an interpolation class to interpolate the bvp
        solution in self.x_opt """

        class LagragePolynomial(object):
            """ Class to interpolate lagrange polynomial with roots at
            t=tau """
            def __init__(self, ts, xs, tau, h):
                self.tau = tau
                self.ts = ts
                self.xs = xs
                self.d  = len(tau)
                self.nk = len(ts)/self.d
                self.h  = h

            def l(self, j, t):
                return np.prod(np.array([ 
                        (t - self.tau[k])/(self.tau[j] - self.tau[k]) 
                        for k in xrange(0,self.d) if k is not j]))

            def interpolate(self, t):
                element_index = int((t/self.h)%self.nk)
                th = (t%self.h)/self.h
                x = self.xs[element_index]
                vals = [self.l(j, th)*x[j] for j in xrange(self.d)] 
                return np.sum(np.array(vals), 0)

            def __call__(self, t):
                try:
                    return np.array([self.interpolate(ti) for ti in t])
                except TypeError:
                    return self.interpolate(t)
            
        return LagragePolynomial(self.tgrid, self.x_opt,
                                 self.coll_setup_dict['tau'], self.h)


    def _bvp_setup(self):
        """ Set up the casadi ipopt solver for the bvp solution """

        # Define some variables
        deg = self.deg
        nk = self.nk
        NV = nk*(deg+1)*self.NEQ

        # NLP variable vector
        V = cs.msym("V", NV)
        XD = np.resize(np.array([],dtype = cs.MX),(nk,deg+1))
        P = cs.msym("P", self.NP*self.nk)
        PK = P.reshape((self.nk, self.NP))

        offset = 0
        for k in range(nk):  
            for j in range(deg+1):
                XD[k][j] = V[offset:offset+self.NEQ]
                offset += self.NEQ

        # Constraint function for the NLP
        g = []
        lbg = []
        ubg = []

        # For all finite elements
        for k in range(nk):
            # For all collocation points
            for j in range(1,deg+1):                
                # Get an expression for the state derivative at the
                # collocation point
                xp_jk = 0
                for j2 in range (deg+1):
                    # get the time derivative of the differential states
                    # (eq 10.19b)
                    xp_jk += self.coll_setup_dict['C'][j2][j]*XD[k][j2]
                
                # Generate parameter set, accounting for variable 
                #
                # Add collocation equations to the NLP
                [fk] = self.rfmod.call([0., xp_jk/self.h, XD[k][j],
                                        PK[k,:].T])

                # impose system dynamics (for the differential states
                # (eq
                # 10.19b))
                g += [fk[:self.NEQ]]
                lbg.append(np.zeros(self.NEQ)) # equality constraints
                ubg.append(np.zeros(self.NEQ)) # equality constraints
            
            # Get an expression for the state at the end of the finite
            # element
            xf_k = 0
            for j in range(deg+1):
                xf_k += self.coll_setup_dict['D'][j]*XD[k][j]
                
            # Add continuity equation to NLP
            # End = Beginning of next
            if k+1 != nk: g += [XD[k+1][0] - xf_k]
            # At the last segment, periodicity constraints
            else: g += [XD[0][0] - xf_k]
            lbg.append(np.zeros(self.NEQ))
            ubg.append(np.zeros(self.NEQ))

        # Nonlinear constraint function
        gfcn = cs.MXFunction([V, P],[cs.vertcat(g)])
        # Objective function (periodicity)
        ofcn = cs.MXFunction([V, P], [cs.sumAll(g[-1]**2)])

        ## ----
        ## SOLVE THE NLP
        ## ----
          
        # Allocate an NLP solver
        self.solver = cs.IpoptSolver(ofcn,gfcn)

        # Set options
        self.solver.setOption("expand_f"         , True)
        self.solver.setOption("expand_g"         , True)
        self.solver.setOption("generate_hessian" , True)
        self.solver.setOption("max_iter"         , 1000)
        self.solver.setOption("tol"              , self.int_opt['bvp_tol'])
        self.solver.setOption("constr_viol_tol"  , self.int_opt['bvp_constr_tol'])
        self.solver.setOption("linear_solver"    , self.int_opt['bvp_linear_solver'])
        self.solver.setOption('parametric'       , True)
        self.solver.setOption('print_level'      , self.int_opt['bvp_print_level'])
        

        # initialize the self.solver
        self.solver.init()
        self.lbg = lbg
        self.ubg = ubg

        self.need_bvp_setup = False

    def bvp_solution(self, pars=None, guess=None, ts=None):
        """ Use a collocation approach to solve for the entrained
        trajectory
        pars - parameters at each collocation point (default - use
               _get_paramset()
        guess - initial guess for xD_init
        ts - t values at which to return solution """

        # Do we need to set up the bvp solver?
        if self.need_bvp_setup: self._bvp_setup()

        # Get parameters over the limit cycle
        if pars is None:
            pars = np.array([self._get_paramset(t) for t in
                             self.tgrid_u]).flatten()

        # Define some variables
        deg = self.deg
        nk = self.nk
        NV = nk*(deg+1)*self.NEQ

        xD_min = self.coll_setup_dict['xD_min']
        xD_max = self.coll_setup_dict['xD_max']

        # Switching to X here.
        if guess is None:
            xD_init = self._calc_initial_guess(pars=pars)[1]
        else:
            assert guess.shape == ((deg+1)*self.nk, self.NEQ),\
                    "Guess shape mismatch: " + guess.shape
            xD_init = guess


        # All variables with bounds and initial guess
        vars_lb   = np.zeros(NV)
        vars_ub   = np.zeros(NV)
        vars_init = np.zeros(NV)
        offset    = 0

        for k in range(nk):  
            # Collocated states
            for j in range(deg+1):
                index = (deg+1)*k + j
                vars_init[offset:offset+self.NEQ] = xD_init[index,:]
                
                vars_lb[offset:offset+self.NEQ] = xD_min
                vars_ub[offset:offset+self.NEQ] = xD_max
                offset += self.NEQ

          
        # Initial condition
        self.solver.setInput(vars_init, cs.NLP_X_INIT)
        self.solver.setInput(pars, cs.NLP_P)

        # Bounds on x
        self.solver.setInput(vars_lb,cs.NLP_LBX)
        self.solver.setInput(vars_ub,cs.NLP_UBX)

        # Bounds on g
        self.solver.setInput(np.concatenate(self.lbg),cs.NLP_LBG)
        self.solver.setInput(np.concatenate(self.ubg),cs.NLP_UBG)

        # Solve the problem
        self.solver.solve()

        success = self.solver.getStat('return_status')
        assert (success == 'Solve_Succeeded'),\
                "Collocation Solve Unsuccessful: " + str(success)
    
        assert (self.solver.output(cs.NLP_COST) < 1E-5), \
                "Collocation Solve reached non-periodic trajectory: " \
                + str(float(self.solver.output(cs.NLP_COST)))

        # Retrieve the solution
        v_opt = np.array(self.solver.output(cs.NLP_X_OPT))
        x_opt = v_opt.reshape((nk,deg+1,self.NEQ))
        x_opt_flat = x_opt.reshape((nk*(deg+1),self.NEQ))

        self.y0 = x_opt_flat[0,:]
        self.x_opt = x_opt
        self.x_opt_flat = x_opt_flat

        if ts is None: return self.tgrid, x_opt_flat
        else:
            interp = self._interpolate_bvp_solution()
            return ts, interp(ts) 

    def calc_arc(self, param, fd=1E-3, plot=False):
        """ Calculate the change in amplitude for each state by changing
        param by amount for each interval """

        try: parind = self.pdict[param]
        except KeyError: parind = param

        # Calculate unforced profiles
        self.bvp_solution()
        unforced_means = self._integrate_xopt()/self.tf
        unforced_stds = (self.x_opt_flat - unforced_means).std(0)

        elements = []
        means = []
        stds = []
        for i in xrange(self.nk):
            try:
                mean, std = self._calc_single_perturbation(i, parind,
                                                           fd)
                means += [mean]
                stds += [std]
                elements += [i]
            except AssertionError: pass
        
        rel_means = (np.array(means) - unforced_means)/unforced_means
        rel_stds = (np.array(stds) - unforced_stds)/unforced_stds
        rel_means *= 1/fd
        rel_stds *= 1/fd
        elements = np.array(elements)

        if plot:
            import matplotlib.pylab as plt
            fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax_mean = axmatrix[0]
            ax_stds = axmatrix[1]
            ax_mean.plot([0, self.tf], [0, 0], '--', color='gray')
            ax_stds.plot([0, self.tf], [0, 0], '--', color='gray')
            ax_mean.plot(elements*self.h, rel_means, '.:')
            ax_stds.plot(elements*self.h, rel_stds, '.:')

        return elements, rel_means, rel_stds


    def _calc_single_perturbation(self, i, parind, fd):
        """ used with calc_arc, calcuates the trajectories for a
        perturbation at a single point """

        pars = np.array([self._get_paramset(t) for t in
                         self.tgrid_u])

        pars[i][parind] *= (1 + fd)
        x_opt_flat = self.bvp_solution(pars=pars.flatten(),
                                       guess=self.x_opt_flat)[1]
        means = self._integrate_xopt()/self.tf
        stds = (x_opt_flat - means).std(0)
        return means, stds

    def _integrate_segment(self, x_segment):
        return np.inner(x_segment, self.coll_setup_dict['E'])*self.h

    def _integrate_xopt(self):
        return np.array([sum([self._integrate_segment(xi) for xi in
                              self.x_opt[:,:,i]]) for i in
                         xrange(self.NEQ)])

    
    def MinimalPerturbation(self, param, initial_strength, tol=1E-1):
        """ Find a sinusoidal perturbation with correct offset for a
        given parameter to ensure that the entrained state trajectories
        resemble the unforced oscillator (t=0 at maximum of state 1).

        Initial_strength = first perturbation guess, must be strong
        enough to quickly entrain oscillator.
        
        Error = Desired error in entrained and unforced trajectories """

        # Support for other forcing profiles? Probably wont be supported
        assert ~bool(self.force_param), "Forcing dictionary must be empty"

        try: par_ind = self.base_class.pdict[param]
        except KeyError: par_ind = int(param)

        ts, guess = self._calc_initial_guess()
        res = np.inf
        phase = 0
        strength = initial_strength

        # Make sure we have the averages
        if not hasattr(self.base_class, 'avg'): self.base_class.average()

        while res > tol:
            phase, res, guess = self._one_perturbation_strength(par_ind, strength, phase, guess)
            if res > tol: strength = 0.1*strength

        self.ForcingParam(param, self._create_oscil(strength, phase))

        return res


    def _create_oscil(self, amp, phase):
        return lambda t: 1 + amp*np.sin(2*np.pi*t/self.tf + phase)
        

    def _one_perturbation_strength(self, par_ind, strength, phase_in, guess):
        """ Helper function to calculate perturbation at one strength.
        Returns optimium phase offset and residual of entrained
        trajectories """

        base = self.base_class

        init_func = self._create_oscil(strength, phase_in)

        yvals = (self.paramset[par_ind] *
                 np.array([init_func(t) for t in self.tgrid_u]))

        pars = np.array([self._get_paramset(tj) for tj in
                         self.tgrid_u])
        pars[:, par_ind] = yvals
        pars = pars.flatten()

        ts, sol = self.bvp_solution(pars=pars, guess=guess)

        def resf(t):
            ret = (sol - base.lc((ts - t)%base.y0[-1]))/base.avg
            return (ret**2).sum()

        tmin = sc.optimize.fmin(resf, self.tf/2., disp=0)[0]
        phase = (phase_in + (tmin/self.tf)*2*np.pi)%(2*np.pi)
        interp = self._interpolate_bvp_solution()
        new_guess = interp((self.tgrid + tmin)%self.tf)

        return (phase, resf(tmin), new_guess)


    def SinglePulseARC(self, param, amount, pulse_duration,
                       trans_duration=3, res=20):
        """ Function to calculate an amplitude response curve based on a
        single perturbation """

        phases = np.linspace(0, 2*np.pi, num=res)
        arc = []
        ts = []

        for phase in phases:
            try: 
                amp = self._single_pulse_comparison(param, phase,
                                                    amount,
                                                    pulse_duration,
                                                    trans_duration)
                arc += [amp/(amount*pulse_duration)]
                ts += [phase*self.base_class.y0[-1]/(2*np.pi)]

            except AssertionError: pass

        return np.array(ts), np.array(arc)

    def _single_pulse_comparison(self, param, phase, amount,
                                 pulse_duration, trans_duration):
        """ Compares a single perturbation to a reference trajectory """

        base = self.base_class
        try: par_ind = self.base_class.pdict[param]
        except KeyError: par_ind = int(param)

        # Use parameterized period so the integration length can be
        # controlled without re-initializing
        self.arcint = cs.CVodesIntegrator(base.modlT)
        self.arcint.setOption('abstol', self.int_opt['int_abstol'])
        self.arcint.setOption('reltol', self.int_opt['int_reltol'])
        self.arcint.setOption('tf', 1.)
        self.arcint.init()

        # Find y0 at start of pulse
        tstart = phase*base.y0[-1]/(2*np.pi)
        tpulse = pulse_duration*base.y0[-1]/(2*np.pi)
        y0 = base.lc(tstart)

        # Integrate trajectory through pulse
        # Find parameter set for pulse
        param_init = np.array(self.paramset)
        param_init[par_ind] *= (1 + amount)

        self.arcint.setInput(y0, cs.INTEGRATOR_X0)
        self.arcint.setInput(param_init.tolist() + [tpulse],
                             cs.INTEGRATOR_P)
        self.arcint.evaluate()
        yf = np.array(self.arcint.output())

        # Simulate the perturbed trajectory for trans_duration.
        tf = base.y0[-1]*trans_duration
        ts = np.linspace(0, tf, num=int(100*trans_duration),
                         endpoint=True)
        self.arcsim = cs.Simulator(self.arcint, ts/tf)
        self.arcsim.init()
        self.arcsim.setInput(yf, cs.INTEGRATOR_X0)
        self.arcsim.setInput(list(base.paramset) + [tf],
                             cs.INTEGRATOR_P)

        self.arcsim.evaluate()
        trajectory = self.arcsim.output().toArray()
        yend = trajectory[-1]
        tend = ts[-1]%base.y0[-1]

        def resy(t):
            return np.linalg.norm(yend - base.lc(t%base.y0[-1]))

        # Minimize resy(t)
        tvals = np.linspace(0, base.y0[-1], num=25)
        tguess = tvals[np.array([resy(t) for t in tvals]).argmin()]
        tmin = sc.optimize.fmin(resy, tguess, disp=0)[0]%base.y0[-1]
        assert resy(tmin)/base.NEQ < 1E-3, "transient not converged"

        tdiff = tmin-tend

        reference = base.lc((ts + tdiff)%base.y0[-1])

        from scipy.integrate import cumtrapz
        amp_change = cumtrapz((trajectory - reference).T, x=ts)[:,-1]

        return amp_change
        
    def _single_pulse_comparison_state(self, state, phase, amount,
                                       trans_duration):
        """ Compares a single perturbation of state to a reference
        trajectory """

        base = self.base_class

        # Use parameterized period so the integration length can be
        # controlled without re-initializing
        self.arcint = cs.CVodesIntegrator(base.modlT)
        self.arcint.setOption('abstol', self.int_opt['int_abstol'])
        self.arcint.setOption('reltol', self.int_opt['int_reltol'])
        self.arcint.setOption('tf', 1.)
        self.arcint.init()

        # Find y0 at start of pulse
        tstart = phase*base.y0[-1]/(2*np.pi)
        y0 = base.lc(tstart)
        y0[state] += amount

        # Simulate the perturbed trajectory for trans_duration.
        tf = base.y0[-1]*trans_duration
        ts = np.linspace(0, tf, num=int(100*trans_duration),
                         endpoint=True)
        self.arcsim = cs.Simulator(self.arcint, ts/tf)
        self.arcsim.init()
        self.arcsim.setInput(y0, cs.INTEGRATOR_X0)
        self.arcsim.setInput(list(base.paramset) + [tf],
                             cs.INTEGRATOR_P)

        self.arcsim.evaluate()
        trajectory = self.arcsim.output().toArray()
        yend = trajectory[-1]
        tend = ts[-1]%base.y0[-1] + tstart

        def resy(t):
            return np.linalg.norm(yend - base.lc(t%base.y0[-1]))

        # Minimize resy(t)
        tvals = np.linspace(0, base.y0[-1], num=25)
        tguess = tvals[np.array([resy(t) for t in tvals]).argmin()]
        tmin = sc.optimize.fmin(resy, tguess, disp=0)[0]%base.y0[-1]
        assert resy(tmin)/base.NEQ < 1E-3, "transient not converged"

        if tmin > base.y0[-1]/2: tmin += -base.y0[-1]

        tdiff = tmin-tend

        # rescale tdiff from -T/2 to T/2
        tdiff = tdiff%base.y0[-1]
        if tdiff > base.y0[-1]/2: tdiff += -base.y0[-1]

        reference = base.lc((ts + tend + tdiff)%base.y0[-1])

        # from scipy.integrate import cumtrapz
        # amp_change = cumtrapz((trajectory - reference).T, x=ts)[:,-1]

        return trajectory, reference, tdiff


        




        
        

         


                








def setup_collocation(tf, nk, deg=5):
    """ Function, possibly moved to utilities, which creates spacing and
            matricies for collocation with desired polynomials """

    # Legendre collocation points
    legendre_points1 = [0,0.500000]
    legendre_points2 = [0,0.211325,0.788675]
    legendre_points3 = [0,0.112702,0.500000,0.887298]
    legendre_points4 = [0,0.069432,0.330009,0.669991,0.930568]
    legendre_points5 = [0,0.046910,0.230765,0.500000,0.769235,0.953090]
    legendre_points = [0, legendre_points1, legendre_points2,
                       legendre_points3, legendre_points4,
                       legendre_points5]

    # Radau collocation points
    radau_points1 = [0,1.000000]
    radau_points2 = [0,0.333333,1.000000]
    radau_points3 = [0,0.155051,0.644949,1.000000]
    radau_points4 = [0,0.088588,0.409467,0.787659,1.000000]
    radau_points5 = [0,0.057104,0.276843,0.583590,0.860240,1.000000]
    radau_points = [0, radau_points1, radau_points2, radau_points3,
                    radau_points4, radau_points5]

    # Type of collocation points
    LEGENDRE = 0
    RADAU = 1
    collocation_points = [legendre_points,radau_points]

    # Radau collocation points
    cp = RADAU
    # Size of the finite elements
    h = float(tf)/nk
     
    # Coefficients of the collocation equation
    C = np.zeros((deg+1,deg+1))
    # Coefficients of the continuity equation
    D = np.zeros(deg+1)
    E = np.zeros(deg+1)

    # Collocation point
    tau = cs.ssym("tau")
      
    # All collocation time points
    tau_root = collocation_points[cp][deg]
    T = np.zeros((nk,deg+1))
    for i in range(nk):
      for j in range(deg+1):
            T[i][j] = h*(i + tau_root[j])

    tg = np.array(tau_root)*h
    for k in range(nk):
        if k == 0:
            tgrid = tg
        else:
            tgrid = np.append(tgrid,tgrid[-1]+tg)
    tgrid_u = np.linspace(0,tf,nk)

    # For all collocation points: eq 10.4 or 10.17 in Biegler's book
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    for j in range(deg+1):
        L = 1
        for j2 in range(deg+1):
            if j2 != j:
                L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
        lfcn = cs.SXFunction([tau],[L])
        lfcn.init()

        lode = cs.SXFunction(cs.daeIn(x=cs.SX('x'), t=tau), cs.daeOut(ode=L))
        lint = cs.CVodesIntegrator(lode)
        lint.setOption('tf', 1.0)
        lint.init()
        lint.setInput(0,0)
        lint.evaluate()
        E[j] = lint.output()

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        lfcn.setInput(1.0)
        lfcn.evaluate()
        D[j] = lfcn.output()
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        for j2 in range(deg+1):
            lfcn.setInput(tau_root[j2])
            lfcn.setFwdSeed(1.0)
            lfcn.evaluate(1,0)
            C[j][j2] = lfcn.fwdSens()

    return_dict = {
        'tgrid'   : tgrid,
        'tgrid_u' : tgrid_u,
        'C'       : C,
        'D'       : D,
        'E'       : E,
        'h'       : h,
        'deg'     : deg,
        'tau'     : tau_root,
    }

    return return_dict





if __name__ == "__main__":
    from CommonFiles.tyson2statemodel import model, paramset, y0in
    from CommonFiles import pBase
    import matplotlib.pylab as plt

    tf = y0in[-1]
    arc_par = 1

    def create_oscil(amp, phase):
        return lambda t: 1 + amp*np.sin(2*np.pi*t/tf + phase)

    def create_square(amp, phase, duration):
        def out(t):
            p = (t*2*np.pi)/tf
            if phase < p < phase+duration:
                return 1.+amp
            else:
                return 1.
        return out


    base = pBase(model(), paramset)
    base.limitCycle()
    forced = ForcedOscil(base, res=200, tf=tf)

    forced.int_opt['bvp_constr_tol'] = 1E-10
    forced.int_opt['bvp_tol'] = 1E-10
    forced.int_opt['bvp_linear_solver'] = 'ma57'
    forced.int_opt['bvp_print_level'] = 0

    # ts, arc = forced.SinglePulseARC(0, 0.5, np.pi/4, res=50)
    # plt.plot(ts, arc[:,0], '.:')

    forced._single_pulse_comparison(0, 0., 0.1, np.pi/4, 2.7)
    # forced.SinglePulseARC(0, 0, 0, 0, 1.5)

    


    # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # ax_mean = axmatrix[0]
    # ax_stds = axmatrix[1]
    # ax_mean.plot([0, forced.tf], [0, 0], '--', color='gray')
    # ax_stds.plot([0, forced.tf], [0, 0], '--', color='gray')

    # print forced.MinimalPerturbation('k1', 0.1, tol=1E-3)
    # elements, rel_means, rel_stds = forced.calc_arc(1, fd=1E-6)
    # ax_mean.plot(elements*forced.h, rel_means[:,0], '.:')
    # ax_stds.plot(elements*forced.h, rel_stds[:,0], '.:')

    # # ts, sol = forced.Integrate(10. * forced.tf, sim=True, res=200)
    # # plt.plot(ts, sol)

    # # forced60 = ForcedOscil(base, res=60, tf=tf)
    # # forced64 = ForcedOscil(base, res=64, tf=tf)
    # # forced = [forced56, forced60, forced64]

    # # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # # ax_mean = axmatrix[0]
    # # ax_stds = axmatrix[1]
    # # ax_mean.plot([0, tf], [0, 0], '--', color='gray')
    # # ax_stds.plot([0, tf], [0, 0], '--', color='gray')
    # # for cl in forced:
    # #     cl.ForcingParam('k1', create_oscil(0.5, np.pi/4 + np.pi), rel=True)
    # #     element, rel_means, rel_stds = cl.calc_arc(1, 1./cl.h)
    # #     ax_mean.plot(element*cl.h, rel_means[:,0], '.:')
    # #     ax_stds.plot(element*cl.h, rel_stds[:,0], '.:')

    # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # ax_mean = axmatrix[0]
    # ax_stds = axmatrix[1]
    # ax_mean.plot([0, tf], [0, 0], '--', color='gray')
    # ax_stds.plot([0, tf], [0, 0], '--', color='gray')

    # means = []
    # stds = []
    # elements = []
    # y0 = base.lc(5*base.y0[-1]/8)

    # for amount in [1.0, 0.5, 0.1, 0.01, 0.001]:
    #     print amount
    #     forced = ForcedOscil(base, res=200, tf=tf)
    #     forced.y0 = y0
    #     forced.int_opt['bvp_constr_tol'] = 1E-10
    #     forced.int_opt['bvp_tol'] = 1E-10
    #     forced.int_opt['bvp_linear_solver'] = 'ma57'
    #     forced.int_opt['bvp_print_level'] = 0
    #     forced.ForcingParam('k1', create_oscil(1E-2, np.pi/4 + np.pi),
    #                         rel=True)
    #     forced.bvp_solution()
    #     t, x = forced.bvp_solution(ts=np.linspace(0, forced.tf, 200))
    #     element, rel_means, rel_stds = forced.calc_arc(arc_par, 1 +
    #                                                    amount)
    #     rel_means = rel_means/amount
    #     rel_stds = rel_stds/amount

    #     means += [rel_means]
    #     stds += [rel_stds]
    #     elements += [element*forced.h]
    #     print forced.solver.output(cs.NLP_COST)
    #     ax_mean.plot(element*forced.h, rel_means[:,0], '.:', label=str(amount))
    #     ax_stds.plot(element*forced.h, rel_stds[:,0], '.:')


    # # for amount in [1.0, 0.1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6]:
    # #     forced = ForcedOscil(base, res=200, tf=60.)
    # #     forced.int_opt['bvp_constr_tol'] = 1E-10
    # #     forced.int_opt['bvp_tol'] = 1E-10
    # #     forced.int_opt['bvp_linear_solver'] = 'ma57'
    # #     forced.int_opt['bvp_print_level'] = 0
    # #     forced.ForcingParam('k1', create_oscil(0.5, np.pi/4 + np.pi), rel=True)
    # #     t, x = forced.bvp_solution(ts=np.linspace(0, forced.tf, 200))
    # #     element, rel_means, rel_stds = forced.calc_arc(arc_par, 1 +
    # #                                                    amount)
    # #     rel_means = rel_means/amount
    # #     rel_stds = rel_stds/amount

    # #     means += [rel_means]
    # #     stds += [rel_stds]
    # #     elements += [element*forced.h]
    # #     print forced.solver.output(cs.NLP_COST)
    # #     ax_mean.plot(element*forced.h, rel_means[:,0], '.:', label=str(amount))
    # #     ax_stds.plot(element*forced.h, rel_stds[:,0], '.:')
    # #     

    # amount = 1E-5
    # forced = ForcedOscil(base, res=200, tf=tf)
    # forced.y0 = y0
    # forced.int_opt['bvp_constr_tol'] = 1E-10
    # forced.int_opt['bvp_tol'] = 1E-10
    # forced.int_opt['bvp_linear_solver'] = 'ma57'
    # forced.int_opt['bvp_print_level'] = 0
    # forced.ForcingParam('k1', create_oscil(1E-2, np.pi/4 + np.pi),
    #                     rel=True)
    # forced.bvp_solution()
    # t, x = forced.bvp_solution(ts=np.linspace(0, forced.tf, 200))
    # element, rel_means, rel_stds = forced.calc_arc(arc_par, 1 +
    #                                                amount)
    # rel_means = rel_means/amount
    # rel_stds = rel_stds/amount

    # means += [rel_means]
    # stds += [rel_stds]
    # elements += [element*forced.h]
    # print forced.solver.output(cs.NLP_COST)
    # ax_mean.plot(element*forced.h, rel_means[:,0], 'k-', label=str(amount))
    # ax_stds.plot(element*forced.h, rel_stds[:,0], 'k-')

    # ax_mean.legend(bbox_to_anchor=(0., 1.02, 1., .102), fancybox=True, ncol=3, mode='expand', borderaxespad=0)

    # elements = np.array(elements)
    # means = np.array(means)
    # stds = np.array(stds)



    # # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # # ax_mean = axmatrix[0]
    # # ax_stds = axmatrix[1]
    # # ax_mean.plot([0, 60.], [0, 0], '--', color='gray')
    # # ax_stds.plot([0, 60.], [0, 0], '--', color='gray')

    # # means = []
    # # stds = []
    # # elements = []

    # # for res in [10, 15, 20, 30, 40, 50, 60]:
    # #     forced = ForcedOscil(base, res=res, tf=60.)
    # #     forced.ForcingParam('k1', create_oscil(0.5, np.pi/4 + np.pi), rel=True)
    # #     t, x = forced.bvp_solution(ts=np.linspace(0, forced.tf, 200))
    # #     element, rel_means, rel_stds = forced.calc_arc(arc_par, 1 + 1E-1/forced.h)
    # #     means += [rel_means]
    # #     stds += [rel_stds]
    # #     elements += [element*forced.h]
    # #     print forced.solver.output(cs.NLP_COST)
    # #     ax_mean.plot(element*forced.h, rel_means[:,0], '.:', label=str(res) + ' nodes')
    # #     ax_stds.plot(element*forced.h, rel_stds[:,0], '.:')
    # #     # plt.plot(t, x)


    # # forced = ForcedOscil(base, res=200, tf=60.)
    # # forced.ForcingParam('k1', create_oscil(0.5, np.pi/4 + np.pi), rel=True)
    # # t, x = forced.bvp_solution(ts=np.linspace(0, forced.tf, 200))
    # # element, rel_means, rel_stds = forced.calc_arc(arc_par, 1 + 1E-1/forced.h)
    # # ax_mean.plot(element*forced.h, rel_means[:,0], 'k-', label='200 nodes')
    # # ax_stds.plot(element*forced.h, rel_stds[:,0], 'k-')

    # # ax_mean.legend(bbox_to_anchor=(0., 1.02, 1., .102), fancybox=True, ncol=4, mode='expand', borderaxespad=0)

    # # elements = np.array(elements)
    # # means = np.array(means)
    # # stds = np.array(stds)



    # # forced.ForcingParam('k1', create_oscil(0.5, np.pi/4 + np.pi), rel=True)
    # # create_square(1.5, 0, np.pi)

    # # plt.figure()
    # # plt.step(forced.tgrid_u, np.array(forced.force_param.values()).T,
    # #          where='post')

 
    # # plt.figure()
    # # for i in xrange(2,40,2):
    # #     ts, sol = forced._calc_initial_guess(periods=i)
    # #     plt.plot(ts, sol[:,1], '.:')

    # # t, x = forced.bvp_solution(guess=sol)
    # 

    # # ts = np.linspace(0, forced.tf, 200)
    # # plt.plot(ts, interp(ts)[:,1], 'k')

    # # ts, sol = forced.Integrate(10 * forced.tf, sim=True, res=200)
    # # plt.figure()
    # # plt.plot(ts, sol)

    # # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # # nodes = []
    # # l = []
    # # for res in xrange(10, 60, 5):
    # #     try:
    # #         tempforced = ForcedOscil(base, res=res, tf=60)
    # #         tempforced.ForcingParam('k1', create_square(2.5, 0, np.pi/2), rel=True)
    # #         elements, rel_means, rel_stds = tempforced.calc_arc(1, 1./tempforced.h)
    # #         ax_mean = axmatrix[0]
    # #         ax_stds = axmatrix[1]
    # #         ax_mean.plot([0, tempforced.tf], [0, 0], '--', color='gray')
    # #         ax_stds.plot([0, tempforced.tf], [0, 0], '--', color='gray')
    # #         ax_mean.plot(elements*tempforced.h, rel_means[:,0], '.:')
    # #         l += [ax_stds.plot(elements*tempforced.h, rel_stds[:,0], '.:')]
    # #         nodes += [res]
    # #     except AssertionError: pass

    # # fig.legend(l, [str(i) + " nodes" for i in nodes], ncol=3)
    # 



    # # plt.legend([str(i) + ' periods' for i in range(2, 10, 2)] + ['bvp'], fancybox=True, loc=3)

    # # plt.plot(ts, sol, '.:')
    # # plt.figure()
    # # plt.step(forced.tgrid_u, np.array(forced.force_param.values()).T, where='post')
    plt.show()

