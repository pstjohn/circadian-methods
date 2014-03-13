import numpy as np
import casadi as cs
import scipy.optimize as opt

from scipy.integrate import trapz

from CommonFiles.pBase import pBase
from CommonFiles.Utilities import (fnlist, plot_grey_zero, PeriodicSpline)


class Amplitude(pBase):
    """ Class to calculate perturbed trajectories and amplitude change
    for a population of oscillators with underlying dynamics described
    by the inhereted pBase class. Intended to replace the older
    AmplitudeResponse class """

    # Shortcut methods
    def _phi_to_t(self, phi): return phi*self.y0[-1]/(2*np.pi)
    def _t_to_phi(self, t): return (2*np.pi)*t/self.y0[-1]

    def lc_phi(self, phi):
        """ interpolate the selc.lc interpolation object using a time on
        (0,2*pi) """
        return self.lc(self._phi_to_t(phi))

    def _create_arc_integrator(self, trans_duration=3, res=100,
                               pulse_res=20):
        """ Create integrator and simulator objects for later use.
        trans_duration is number of periods to simulate perturbed
        trajectory and reference, res is the resolution (per period) of
        the output trajectories """

        # Use parameterized period so the integration length can be
        # controlled without re-initializing
        self.arcint = cs.CVodesIntegrator(self.modlT)
        self.arcint.setOption('abstol', self.intoptions['int_abstol'])
        self.arcint.setOption('reltol', self.intoptions['int_reltol'])
        self.arcint.setOption('tf', 1.)
        self.arcint.init()
        #
        # Simulate the perturbed trajectory for trans_duration.
        tf = self.y0[-1]*trans_duration
        traj_res = int(res*trans_duration)
        self.arc_traj_ts = np.linspace(0, tf, num=traj_res,
                                       endpoint=True) 
        self.arcsim = cs.Simulator(self.arcint, self.arc_traj_ts/tf)
        self.arcsim.init()
        self.pulsesim = cs.Simulator(self.arcint,
                                     np.linspace(0, 1., num=pulse_res,
                                                 endpoint=True))
        self.pulsesim.init()

    def _simulate_pulse_and_ref(self, pulse, trans_duration):
        """ Simulate the pulse and match with the limit cycle to find
        the amplitude and phase response. Returns (trajectory,
        reference, time offset)"""

        self.arcsim.setInput(pulse['x0'], cs.INTEGRATOR_X0)
        self.arcsim.setInput(list(self.paramset) +
                             [self.y0[-1]*trans_duration],
                             cs.INTEGRATOR_P)
        self.arcsim.evaluate()
        trajectory = self.arcsim.output().toArray()

        yend = trajectory[-1]
        tend = (self.arc_traj_ts[-1]%self.y0[-1] +
                self._phi_to_t(pulse['phi']))

        def resy(t):
            return np.linalg.norm(yend - self.lc(t%self.y0[-1]))

        # Minimize resy(t)
        tvals = np.linspace(0, self.y0[-1], num=25)
        tguess = tvals[np.array([resy(t) for t in tvals]).argmin()]
        tmin = opt.fmin(resy, tguess, disp=0)[0]%self.y0[-1]
        assert resy(tmin)/self.NEQ < 1E-3, "transient not converged"

        if tmin > self.y0[-1]/2: tmin +=-self.y0[-1]

        tdiff = tmin-tend

        # rescale tdiff from -T/2 to T/2
        tdiff = tdiff%self.y0[-1]
        if tdiff > self.y0[-1]/2: tdiff += -self.y0[-1]

        reference = self.lc((self.arc_traj_ts + tend + tdiff)%self.y0[-1])

        return_dict = {
            'traj'   : trajectory,
            'ref'    : reference,
            'p_diff' : self._t_to_phi(tdiff)}

        if pulse['type'] is 'param':
            return_dict['pulse_traj'] = pulse['traj']
            return_dict['pulse_ts']   = pulse['ts']
            
        return return_dict

    def _s_pulse_creator(self, state, amount):
        """ Compares a single perturbation of state to a reference
        trajectory """

        def pulse_creator(phi):
            x0 = self.lc_phi(phi)
            x0[state] += amount
            return {
                'type' : 'state',
                'x0'   : x0,
                'phi'  : phi,
            }

        return pulse_creator

    def _p_pulse_creator(self, par_ind, amount, pulse_duration):
        """ Compares a single perturbation of parameter at phase phi_start
        for pulse_duration (radians) to a reference trajectory """

        def pulse_creator(phi):
            # Find conditions at start of pulse
            x_start = self.lc_phi(phi - pulse_duration)
            param_init = np.array(self.paramset)
            param_init[par_ind] += amount

            # Integrate trajectory through pulse
            # Find parameter set for pulse
            self.pulsesim.setInput(x_start, cs.INTEGRATOR_X0)
            self.pulsesim.setInput(param_init.tolist() +
                                 [self._phi_to_t(pulse_duration)],
                                 cs.INTEGRATOR_P)
            self.pulsesim.evaluate()
            pulse_trajectory = np.array(self.pulsesim.output())
            x0 = pulse_trajectory[-1]
            
            pulse = {
                'type' : 'param',
                'x0' : x0,
                'phi' : phi,
                'traj' : pulse_trajectory,
                'ts' : np.linspace(self._phi_to_t(-pulse_duration), 0,
                                   num=len(pulse_trajectory),
                                   endpoint=True),
            }
            return pulse

        return pulse_creator

    def _calc_amp_change(self, ts, trajectory, reference, mean=None):
        """ Calculate the amplitude change between trajectorys and an
        asympotitic reference. Each may be the trajectories of several
        states """

        if mean is None: mean = self.avg

        from scipy.interpolate import UnivariateSpline

        h = ((trajectory - mean)**2 - (reference - mean)**2)
        h_inds = xrange(h.shape[1])

        h_interp_list = [UnivariateSpline(ts, h[:,i], s=0) for i in
                         h_inds]

        return np.array([h_interp_list[i].integral(0, ts[-1]) for i in
                         h_inds])

    def calc_pulse_responses(self, pulse_creator, trans_duration=3,
                             res=100):
        """ Integrate pulses starting at different initial phases to
        find x(phi, t), create interpolation object, and find amplitude
        and phase response curves for the given pulse """

        self._create_arc_integrator(trans_duration)
        if not hasattr(self, 'avg') : self.average()
        if not hasattr(self, 'lc')  : self.limitCycle()

        pulse = pulse_creator(0.)

        phis = np.linspace(0, 2*np.pi, num=res)
        trajectories = []
        references = []
        arc = []
        prc = []

        if pulse['type'] is 'param':
            pulse_traj = []

        for phi in phis:
            out = self._simulate_pulse_and_ref(pulse_creator(phi),
                                               trans_duration)
            trajectories += [out['traj']]
            references += [out['ref']]
            prc += [out['p_diff']]
            arc += [self._calc_amp_change(self.arc_traj_ts, out['traj'],
                                          out['ref'])]

            if pulse['type'] is 'param':
                pulse_traj += [out['pulse_traj']]

        trajectories = np.array(trajectories) 
        references = np.array(references)

        self.traj_interp = cyl_interp(phis, self.arc_traj_ts,
                                       trajectories)
        self.ref_interp = cyl_interp(phis, self.arc_traj_ts,
                                       references)

        if pulse['type'] is 'param':
            pulse_traj = np.array(pulse_traj)
            comb_ts = np.hstack([pulse['ts'][:-1], self.arc_traj_ts])
            comb_traj = np.concatenate([pulse_traj[:,:-1,:],
                                        trajectories], axis=1)
            self.comb_interp = cyl_interp(phis, comb_ts, comb_traj)

        self.pulse_creator = pulse_creator
        self.prc_single_cell = np.array(prc) # Single Cell PRC
        self.arc_single_cell = np.array(arc) # Single Cell ARC
        self.phis = phis

        self.prc_interp = PeriodicSpline(self.phis,
                                         self.prc_single_cell,
                                         period=2*np.pi)
        
        self.ptc_interp = lambda x: x + self.prc_interp(x)



    def calc_population_responses(self, input_phase_distribution,
                                  tarc=True):
        """ Calculate the population level PRC and ARC resulting from
        the perturbation in self.pulse_creator on the population with
        std sigma """

        self.phase_distribution = input_phase_distribution

        g_phis = self.phis + self.prc_single_cell

        # Create 
        pd = phase_distribution(0., self.phase_distribution.sigma(0))
        # pdf with new mean at each phase
        pdfs = pd(self.phis, self.phis) 

        z_bar = trapz(np.exp(1j*self.phis) * pdfs, x=self.phis)
        z_hat = trapz(np.exp(1j*g_phis) * pdfs, x=self.phis)

        angle_0, length_0 = mean_length(z_bar)
        angle_f, length_f = mean_length(z_hat)

        self.prc_population = normalize(angle_f - angle_0)
        self.arc_population = length_f - length_0


        if tarc:
            self.tarc_population = []

            for phi in self.phis:
                phi_o = phi - self.phase_distribution.u0
                trans = self.x_hat(self.arc_traj_ts, phi_offset=phi_o)
                ref = self.x_hat_ss(self.arc_traj_ts, phi_offset=phi_o)
                self.tarc_population += [
                    self._calc_amp_change(self.arc_traj_ts, trans, ref)]

            self.tarc_population = np.array(self.tarc_population)
    
    def x_bar(self, ts):
        """ Function to return the average expression level as a
        function of the current phase distribution """

        ts_phi = self._t_to_phi(ts)
        return self.phase_distribution.integrate(self.phis, ts_phi,
                                                 self.lc_phi(self.phis))


    def x_hat(self, ts, phi_offset=0):
        """ Function to return the average expression level after
        perturbation for the given initial phase distribution and pulse
        """

        try:
            # See if a pulse interpolator exists
            trajectories=self.comb_interp(self.phis, ts)
        except AttributeError:
            trajectories = self.traj_interp(self.phis, ts)

        pdf_single = self.phase_distribution(self.phis, phi_offset)[:,None]
        pdf = np.tile(pdf_single, self.NEQ)
        return trapz(trajectories * pdf, x=self.phis,
                     axis=trajectories.ndim - 2)


    def x_hat_ss(self, ts, phi_offset=0):
        """ Calculates the steady-state final mean phase oscillation
        using calculated reference trajectories """
        
        # references = self.ref_interp(self.phis, ts)

        references = np.array([self.lc(ts +
                                       self._phi_to_t(self.ptc_interp(phi)))
                                       for phi in self.phis]).swapaxes(1,0)
        pdf_single = self.phase_distribution(self.phis, phi_offset)[:,None]
        pdf = np.tile(pdf_single, self.NEQ)
        return trapz(references * pdf, x=self.phis, axis=1)

            

        
class phase_distribution(object):        
    def __init__(self, u0, sigma_0, sigma_decay=0.):
        """ class to create a function describing the evolution of a
        population of oscillators with mean phase u0 (at t=0) and
        initial standard deviation sigma_0 (increases linearly with time
        according to sigma_decay). Should return a callable function
        f(phi, t) to describe the probability density function (t in
        phase space) """

        def lin_with_bound(t):
            t = np.atleast_1d(t)
            change = np.max(np.vstack([np.zeros(t.shape),
                                       sigma_decay*t]), 0)
            return sigma_0 + change

        self.sigma = lin_with_bound
        self.u0 = u0

    def __call__(self, phis, ts):
        """ Evaluate the distribution at the desired time. T should be
        in phase space """

        ts = np.atleast_1d(ts)

        try: sigmas = self.sigma(ts)
        except TypeError: sigmas = n*[self.sigma]

        return np.array([wrapped_gaussian(phis, (self.u0 + t)%(2*np.pi),
                sigma) for t, sigma in zip(ts, sigmas)]).squeeze()


    def integrate(self, phis, ts, values):
        """ Function to evaluate the average value of "values" over the
        desired phases at time t """

        try: len(ts)
        except TypeError: ts = [ts]

        pdfs = np.tile(self(phis, ts)[:,:,None], values.shape[1])
        return trapz(values * pdfs, x=phis, axis=1)



def wrapped_gaussian(phis, mu, sigma, abstol=1E-10):
    """ A wrapped gaussian defined on 2->2Pi, with mean mu and
    standard deviation sigma. Thetas are desired output angles.
    Chooses more efficient method """

    def wrapped_gaussian_fourier(phis, mu, sigma, abstol=1E-10):
        """ A wrapped gaussian defined on 2->2Pi, with mean mu and
        standard deviation sigma. Thetas are desired output angles.
        Efficient at high standard deviations """

        tsum = 1/(2*np.pi)*np.ones(len(phis))

        for i in xrange(1, 1000): # Maximum iterations
            oldtsum = np.array(tsum)
            tsum += np.exp(-i**2*sigma**2/2)*np.cos(i*(phis - mu))/np.pi

            diff = (tsum - oldtsum)

            if diff.max() < abstol: break

        return tsum

    def wrapped_gaussian_direct(phis, mu, sigma, abstol=1E-10):
        """ A wrapped gaussian defined on 2->2Pi, with mean mu and
        standard deviation sigma. Thetas are desired output angles.
        Efficient at low standard deviations """

        norm = 1/(sigma*np.sqrt(2*np.pi))

        tsum = np.zeros(len(phis))
        for i in xrange(1000): # Maximum iterations
            oldtsum = np.array(tsum)

            # add both +/- i
            if i == 0: 
                tsum += norm*np.exp(-((phis - mu)**2)/(2*sigma**2))
            
            else:
                tsum += norm*np.exp(-((phis - mu +
                                       2*np.pi*i)**2)/(2*sigma**2))
                tsum += norm*np.exp(-((phis - mu -
                                       2*np.pi*i)**2)/(2*sigma**2))
            
            diff = (tsum - oldtsum)

            if diff.max() < abstol: break

        return tsum

    if sigma > 1.4:
        return wrapped_gaussian_fourier(phis, mu, sigma, abstol)
    else: return wrapped_gaussian_direct(phis, mu, sigma, abstol)

def mean_length(z):
    """ Return the mean angle and resultant length of the complex
    variable z """

    return (np.angle(z), np.abs(z))

def normalize(angles, end=np.pi):
    """ normalize angles to the range -Pi, Pi """

    angles = angles % (2*np.pi)
    angles[angles > end] += -2*np.pi

    return angles




class cyl_interp(object):
    """ Class to interpolate x(phi, t) objects and store relevant data
    and methods for later use """

    def __init__(self, phis, ts, input_matrix):
        """ Input matrix should contain data in (phi, t, n) format,
        where phi is the phase, t is the time, and n is the state index
        """

        self.ts = ts
        self.phis = phis

        if (phis[-1] % (2*np.pi)) != phis[0]:
            phis = np.hstack([phis, phis[0]+2*np.pi])
            input_matrix = np.concatenate([input_matrix,
                           input_matrix[0,:,:].reshape(1,300,2)])

        from scipy.interpolate import RectBivariateSpline
        self.interp_object = [RectBivariateSpline(phis, ts,
                                                  input_matrix[:,:,i])
                              for i in xrange(input_matrix.shape[2])]

        self.interp_object = fnlist(self.interp_object)

    def __call__(self, phi, t):
        """ Evaluate spline at the grid points defined by the coordinate
        arrays phi, t. """

        return self.interp_object(phi, t).squeeze().T












if __name__ == "__main__":

    from CommonFiles.Models.tyson2statemodel import model, paramset
    import matplotlib.pylab as plt
    from CommonFiles.PlotOptions import (PlotOptions, format_2pi_axis,
                                         layout_pad)

    PlotOptions(uselatex=True)

    test = Amplitude(model(), paramset)
    # state_pulse_creator = test._s_pulse_creator(0, 0.1)
    param = 4
    amount = 0.25
    duration = np.pi/2
    state_pulse_creator = test._p_pulse_creator(param, amount, duration)
    test.calc_pulse_responses(state_pulse_creator)

    # # Fig 1 : Test single cell PRC and ARC
    # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # axmatrix[0].plot(test.phis, test.prc_single_cell)
    # axmatrix[0].set_title('Single Cell PRC')
    # axmatrix[1].plot(test.phis, test.arc_single_cell[:,0])
    # axmatrix[1].set_title('Single Cell ARC')
    # plot_grey_zero(axmatrix[0])
    # plot_grey_zero(axmatrix[1])
    # format_2pi_axis(axmatrix[1])

    test_population = phase_distribution(0.5*np.pi/2, 0.5, 0.1)
    test.calc_population_responses(test_population, tarc=False)

    std_single = test.arc_single_cell[:,0]/test.arc_single_cell[:,0].std()
    # std_pop = test.tarc_population[:,0]/test.arc_single_cell[:,0].std()
    std_desynch = test.arc_population/test.arc_population.std()

    # get differential amplitude ARC and PRC
    # test.findARC_whole()

    fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    axmatrix[0].plot(test.phis, test.prc_single_cell, label='Single Cell')
    axmatrix[0].plot(test.phis, test.prc_population, label='Population')
    # axmatrix[0].plot(test._t_to_phi(test.prc_ts),
    #                  amount*duration*test.pPRC[:,param],
    #                  ':', label='Differential')
    axmatrix[0].set_title('PRC')
    axmatrix[0].legend(loc='best')
    axmatrix[0].set_ylabel(r'$\Delta\phi$')
    axmatrix[1].plot(test.phis, std_single, label='Single Cell')
    # axmatrix[1].plot(test.phis, std_pop, label='Population')
    # axmatrix[1].plot(test._t_to_phi(test.prc_ts),
    #                  (amount*test._phi_to_t(duration)*test.pARC[:,0,param]
    #                   / test.arc_single_cell[:,0].std()), ':',
    #                  label='Differential')
    axmatrix[1].plot(test.phis, std_desynch, '--', label='Desynch')
    axmatrix[1].set_title('ARC')
    axmatrix[1].set_xlabel(r'$\phi$')
    axmatrix[1].legend(loc='best')
    plot_grey_zero(axmatrix[0])
    plot_grey_zero(axmatrix[1])
    format_2pi_axis(axmatrix[1])
    fig.tight_layout(**layout_pad)


    t0 = test._phi_to_t(test.phase_distribution.u0)
    ts_ref = np.linspace(-test.y0[-1], 3*test.y0[-1], 180)
    try: ts = test.comb_interp.ts
    except AttributeError: ts = test.traj_interp.ts
    xbar = test.x_bar(ts_ref)
    xhat = test.x_hat(ts)

    # Fig 3 : Test x(t) functions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts_ref, xbar[:,0], ':', label=r'$\bar{x}(t)$')
    ax.plot(ts[10:], test.x_hat_ss(ts[10:])[:,0], '-',
            label=r'$\hat{x}_{\text{ss}}(t)')
    ax.plot(ts, xhat[:,0], 'r--', label=r'$\hat{x}(t)$')
    ax.legend(loc=2)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    fig.tight_layout(**layout_pad)

    # Fig. 4: State Space evolution
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x0 = test.x_hat(0)
    # xhat_ss = test.x_hat_ss(test._phi_to_t(test.phis))
    # xbar = test.x_bar(test._phi_to_t(test.phis))
    # ax.plot(xhat[:,0], xhat[:,1], label=r'$\hat{x}(t)$')
    # ax.plot(xbar[:,0], xbar[:,1], '--', label=r'$\bar{x}(t)$')
    # ax.plot(xhat_ss[:,0], xhat_ss[:,1], 'k',
    #         label=r'$\hat{x}_{\text{ss}}(t)')
    # ax.plot(x0[0], x0[1], 'ro')





    plt.show()
