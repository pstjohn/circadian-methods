import numpy as np
import casadi as cs
import scipy.optimize as opt

from math import sqrt
from scipy.special import erf

from scipy.interpolate import UnivariateSpline

from CommonFiles.pBase import pBase
from CommonFiles.Utilities import (fnlist, plot_grey_zero,
                                   PeriodicSpline,
                                   ComplexPeriodicSpline, p_integrate,
                                   ptc_from_prc, RootFindingSpline)


class Amplitude(pBase):
    """ Class to calculate perturbed trajectories and amplitude change
    for a population of oscillators with underlying dynamics described
    by the inhereted pBase class. Intended to replace the older
    AmplitudeResponse class 
    
    I really should have made this class work underneath the population
    distribution classes, where a population of oscillators can have
    perturbations applied to them at arbitrary times, and this class
    calculates the relevant phase-dependent changes necessary to change
    the distribution. Would take a bunch of reformatting, so perhaps I
    will pick this up at a later time.
    """

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
        tf = 2*np.pi*trans_duration
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

        yend = trajectory[-5:] # Take the last 5 points
        phi_end = (self.arc_traj_ts[-1] + pulse['phi'])%(2*np.pi)
        phi_diff = self.arc_traj_ts[-1] - self.arc_traj_ts[-2]
        phi_step_arr = np.arange(-4, 1) * phi_diff

        def resy(phi):
            return np.linalg.norm(yend - 
                                  self.lc_phi(phi + phi_step_arr))

        # Minimize resy(t)
        phi_vals = np.linspace(0, 2*np.pi, num=25)
        phi_guess = phi_vals[np.array([resy(phi) for phi in
                                       phi_vals]).argmin()]
        p_min = opt.fmin(resy, phi_guess, disp=0)[0]%(2*np.pi)
        assert resy(p_min)/self.avg.sum() < 1E-3, "transient not converged"


        phi_diff = normalize(p_min - phi_end)

        reference = self.lc_phi(self.arc_traj_ts + phi_end + phi_diff)

        return_dict = {
            'traj'   : trajectory,
            'ref'    : reference,
            'p_diff' : phi_diff}

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
        
        assert pulse_duration > 0, "Pulse duration must be positive"

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
                'ts' : np.linspace(-pulse_duration, 0,
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

        h = ((trajectory - mean)**2 - (reference - mean)**2)
        h_inds = xrange(h.shape[1])

        h_interp_list = [UnivariateSpline(ts, h[:,i], s=0) for i in
                         h_inds]

        return np.array([h_interp_list[i].integral(0, ts[-1]) for i in
                         h_inds])

    def _init_amp_class(self, trans_duration=3, res=100):
        """ Sets up some important class variables. Useful if pulse
        responses at each phase is not needed """

        self._create_arc_integrator(trans_duration)
        if not hasattr(self, 'avg') : self.average()
        if not hasattr(self, 'lc')  : self.limitCycle()
        phis = np.linspace(0, 2*np.pi, num=res, endpoint=True)
        self.phis = phis


    def calc_pulse_responses(self, pulse_creator, trans_duration=3,
                             res=100):
        """ Integrate pulses starting at different initial phases to
        find x(phi, t), create interpolation object, and find amplitude
        and phase response curves for the given pulse """

        self._init_amp_class(trans_duration, res)

        pulse = pulse_creator(0.)

        trajectories = []
        references = []
        arc = []
        prc = []

        if pulse['type'] is 'param':
            pulse_traj = []

        for phi in self.phis:
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

        self.traj_interp = cyl_interp(self.phis, self.arc_traj_ts,
                                       trajectories)
        self.ref_interp = cyl_interp(self.phis, self.arc_traj_ts,
                                       references)

        if pulse['type'] is 'param':
            pulse_traj = np.array(pulse_traj)
            comb_ts = np.hstack([pulse['ts'][:-1], self.arc_traj_ts])
            comb_traj = np.concatenate([pulse_traj[:,:-1,:],
                                        trajectories], axis=1)
            self.comb_interp = cyl_interp(self.phis, comb_ts, comb_traj)

        self.pulse_creator = pulse_creator
        self.prc_single_cell = np.array(prc) # Single Cell PRC
        self.arc_single_cell = np.array(arc) # Single Cell ARC

        self.prc_interp = PeriodicSpline(self.phis,
                                         self.prc_single_cell,
                                         period=2*np.pi)
        
        self.ptc_interp = ptc_from_prc(self.prc_interp)

    def _z_from_induced_prc(self):
        """ Given a single PRC in self.prc_single_cell and phase
        distribution in self.phase_distribution, calculate the resulting
        mean phase distributions from a pulse given at any phase.
        Returns interpolating objects for z_bar and z_hat, the
        unperturbed and perturbed complex variables """

        phis = self.phis
        g_phis = phis + self.prc_single_cell

        # Create 
        pdfs = self.phase_distribution.phase_offset(phis, phis) 

        z_bar = p_integrate(phis, (np.exp(1j*phis) * pdfs).T)
        z_hat = p_integrate(phis, (np.exp(1j*g_phis) * pdfs).T)

        self.z_bar = ComplexPeriodicSpline(phis, z_bar)
        self.z_hat = ComplexPeriodicSpline(phis, z_hat)


    def calc_population_responses(self, input_phase_distribution,
                                  tarc=False):
        """ Calculate the population level PRC and ARC resulting from
        the perturbation in self.pulse_creator on the population with
        std sigma """

        self.phase_distribution = input_phase_distribution
        self._z_from_induced_prc()

        angle_0, length_0 = mean_length(self.z_bar(self.phis))
        angle_f, length_f = mean_length(self.z_hat(self.phis))



        self.prc_population = normalize(angle_f - angle_0)
        self.arc_population = length_f - length_0


        if tarc:
            self.tarc_population = []

            for phi in self.phis:
                phi_o = phi - self.phase_distribution.mu
                trans = self.x_hat(self.arc_traj_ts, phi_offset=phi_o)
                ref = self.x_hat_ss(self.arc_traj_ts, phi_offset=phi_o)
                self.tarc_population += [
                    self._calc_amp_change(self.arc_traj_ts, trans, ref)]

            self.tarc_population = np.array(self.tarc_population)
    
    def x_bar(self, ts, phi_offset=0):
        """ Function to return the average expression level before
        perturbation as a function of the current phase distribution """

        pd = self.phase_distribution
        lc = self.lc_phi(self.phis + phi_offset)
        return pd.average(ts, lc, self.phis).T

    def x_hat(self, ts, phi_offset=0, approx=False):
        """ Function to return the approximate actual expression level
        after perturbation at the given phi_offset. Uses the inverted
        pdf (approx=False) or a gaussian approximation (approx=True) """

        ts = np.atleast_1d(ts)

        # Find after-pulse values
        after = ts >= 0
        traj_out = np.zeros((len(ts), self.NEQ))

        # Calculate the steady-state trajectory after the pulse (only
        # valid after pulse)
        x_hat_ss = self.x_hat_ss(ts[after], phi_offset, approx=approx)
        traj_out[after] = x_hat_ss

        # Calculate the initial trajectory prior to the pulse (only for
        # parameter perturbation
        pd = self.phase_distribution

        # Evaluate \Delta x(t), the difference between the reference and
        # transient trajectories (Again, only valid after the pulse)

        adj_phis = self.phis + phi_offset
        try: 
            Delta_x = (self.comb_interp(adj_phis, ts[after]) -
                       self.ref_interp(adj_phis, ts[after]))

            # Evaluate the pre-pulse transients (for finite-duration
            # parameter perturbations
            if not np.all(after): # Only if we need negative times
                x_before = self.comb_interp(self.phis, ts[~after])
                pdf = pd.phase_offset(self.phis, phi_offset)[:,None]
                avg_xb = (pdf*x_before).swapaxes(1,0)
                traj_out[~after] = p_integrate(self.phis, avg_xb).T

        except AttributeError:
            Delta_x = (self.traj_interp(adj_phis, ts) -
                       self.ref_interp(adj_phis, ts))

        pdf = pd(ts[after], advance_t=False)[:,:,None]
        Averaged_dx = p_integrate(self.phis,
                                  (pdf*Delta_x).swapaxes(1,0)).T

        traj_out[after] += Averaged_dx

        return traj_out


    def x_hat_ss(self, ts, phi_offset=0, approx=False):
        """ Calculates the steady-state final mean phase oscillation by
        inverting the phase transition curve (approx=False) or by
        finding the gaussian approximation for the perturbation
        (approx=True)"""
        
        pd = self.phase_distribution
        if approx:
            mean_p, std_p = mean_std(self.z_hat(phi_offset))
            pt = gaussian_phase_distribution(mean_p, std_p,
                                             pd.phase_diffusivity)
        else:
            pt = pd.invert(self.phis, self.prc_single_cell,
                           phi_offset=phi_offset)

        return pt.average(ts, self.lc_phi(self.phis), self.phis).T


    def _cos_components(self):
        """ return the phases and amplitudes associated with the first
        order fourier compenent of the limit cycle (i.e., the best-fit
        sinusoid which fits the limit cycle) """
    
        dft_sol = np.fft.fft(self.sol[:-1], axis=0)
        n = len(self.ts[:-1])
        baseline = dft_sol[0]/n
        comp = 2./n*dft_sol[1]
        return np.abs(comp), np.angle(comp), baseline




        
class phase_distribution(object):        
    def __init__(self, fo_phi, phase_diffusivity,
                 invert_res=70):
        """ class to create a function describing the evolution of a
        population of oscillators with initial normalized phase
        distribution fo_phi, period, and phase diffusivity. Contains
        methods to evaluate the probability distribtion at specified phi
        and t, as well as find the mean value of functions """

        self.fo_phi = fo_phi
        self.period = 2*np.pi
        self.phase_diffusivity = phase_diffusivity
        self.invert_res = invert_res

        phis = np.linspace(0, 2*np.pi, 100)
        pdf = self.fo_phi(phis)
        z_bar = p_integrate(phis, np.exp(1j * phis)*pdf)
        self.mu, self.length = mean_length(z_bar)
        self.sigma = np.sqrt(-2*np.log(self.length))
        


    def __call__(self, ts, phi_res=100):
        """ Evaluate the distribution at the desired time and phases.
        phi_res is resolution of output phis, and ts should be the same
        units as period. Returns an array """

        # Set up time variables
        ts = np.atleast_1d(ts)
        phis = np.linspace(0, 2*np.pi, num=phi_res, endpoint=True)
        phi_diff = phis[1]

        # Find original distribution fo(phi)
        fo = self.fo_phi(phis)

        # Return distribution matrix, evaluated at phis and ts
        ret_phis = np.zeros((len(ts), phi_res))

        # For every time point, find distribution
        for i, t in enumerate(ts): ##Parallelize?
            sigma = np.sqrt(2*self.phase_diffusivity*t) if t > 0 else 0.
            mean = (t * (2*np.pi)/self.period)%(2*np.pi)
            if sigma > phi_diff: 
                # Convolute with gaussian
                normal = wrapped_gaussian(phis, mean, sigma)
                # Remove last data point before dft, return after.
                temp = np.zeros(fo.shape)
                temp[:-1] = convolve_pdf(fo[:-1], normal[:-1], phi_diff)
                temp[-1] = temp[0]
                ret_phis[i] = temp

            else: 
                # Close to delta function, use original distribution
                ret_phis[i] = self.fo_phi(phis - mean)

        return ret_phis

    def phase_offset(self, phis, phi_offsets):
        """ Shift the initial conditions by phi_offsets, used in
        constructing amplitude response curve for the population of
        oscillators """

        phi_offsets = np.atleast_1d(phi_offsets)
        out_arr = np.zeros((len(phi_offsets), len(phis)))
        for i, phi_o in enumerate(phi_offsets):
            out_arr[i] = self.fo_phi((phis - phi_o)%(2*np.pi))
        return out_arr.squeeze()

    def average(self, ts, values, phis):
        """ Function to evaluate the average value of values(phis) over the
        desired phases at time t. Values/phis should be equally spaced
        over the interval, and not include the endpoint """

        # Check inputs
        ts = np.atleast_1d(ts)
        values = np.atleast_2d(values)
        if values.shape[0] != len(phis): values = values.T

        pdfs = self(ts, len(phis))
        expanded_values = np.tile(values[:,:,None], len(ts))
        expanded_pdfs = np.tile(pdfs[:,:,None], values.shape[1])
        expanded_pdfs = expanded_pdfs.swapaxes(1,2).swapaxes(0,2)

        return p_integrate(phis, expanded_values * expanded_pdfs).T

    def invert(self, phis, prc, phi_offset=0):
        """ Find the resulting probability density function after
        applying a perturbation, resulting in phase response curve prc,
        at time 0. Returns a phase_distribution class of the perturbed
        system. """

        prc_interp = PeriodicSpline(phis, prc)
        ptc_interp = ptc_from_prc(prc_interp)

        res = self.invert_res
        iphis = np.linspace(0, 2*np.pi, num=res, endpoint=True)
        spacing = iphis[1]
        x = np.linspace(-2*np.pi, 4*np.pi, num=3*res, endpoint=True) 

        # Is the phase transition curve monotonic?
        # if so, use abbreviated formula
        if ptc_interp(phis, 1).min() > 1E-2:
            invertedptc = UnivariateSpline(ptc_interp(x), x, s=0)
            f_perturbed = (invertedptc(iphis, 1) *
                           self.phase_offset(invertedptc(iphis),
                                             phi_offset))

        # Not monotonic, must calculate roots at each point
        else: 
            f_perturbed = np.zeros(res)
            root_interp = RootFindingSpline(x, ptc_interp(x))
     
            # import matplotlib.pylab as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(x, root_interp(x))

            for i,phi in enumerate(iphis):
                start = phi - spacing/2
                stop  = phi + spacing/2
                roots_above = root_interp.root_offset(stop)
                roots_below = root_interp.root_offset(start)
                roots = np.hstack([roots_below, roots_above])
                roots.sort()
                roots = roots.reshape((-1, 2))
                for pair in roots:
                    # ax.fill_between([pair[0], pair[1]], start, stop,
                    # alpha=0.3)
                    # dx = pair[1] - pair[0]
                    dy = spacing
                    int_fx = self.integrate(pair[0], pair[1],
                                            phi_offset)
                    f_perturbed[i] += int_fx/dy


        scale = p_integrate(iphis, f_perturbed)
        assert scale > 0.975, \
        "pdf inversion error (Error = %0.3f)"%scale
        f_perturbed *= 1./scale 

        f_p_interp = PeriodicSpline(iphis, f_perturbed)

        return phase_distribution(f_p_interp, self.phase_diffusivity,
                                  invert_res=self.invert_res)


    def integrate(self, a, b, phi_offset=0):
        return self.fo_phi.integrate(a-phi_offset, b-phi_offset)



class gaussian_phase_distribution(phase_distribution):
    """ Class to handle the specific case where the phase distribution
    is a wrapped gaussian function, with methods that take into account
    the more tractable nature of gaussian distributions. """

    def __init__(self, mu, sigma, phase_diffusivity,
                 invert_res=60):
        """ mu and sigma should be the mean and standard deviation of
        the trajectory. """

        assert np.isfinite(mu), "Mu must be a number"
        assert np.isfinite(sigma), "Sigma must be a number"
        assert np.isfinite(phase_diffusivity), "D must be a number"
        
        # General phase distribution variables
        self.phase_diffusivity = phase_diffusivity
        self.period = 2*np.pi
        self.invert_res = invert_res

        # Gaussian specific variables
        self.mu = mu%(2*np.pi)
        self.sigma = sigma
        self.length = np.exp((self.sigma**2)/(-2))


        self.fo_phi = lambda phis: wrapped_gaussian(phis%(2*np.pi), mu,
                                                    sigma)
        phis = np.linspace(0, 2*np.pi, 100)
        self.fo_phi_interp = PeriodicSpline(phis, self.fo_phi(phis))

    def __call__(self, ts, phi_res=100, advance_t=True):
        """ Convolute the original gaussian with the diffusion update to
        generate the pdf at time t. Advance_t=False hold the mean in
        place for x_hat """

        # Set up time variables
        ts = np.atleast_1d(ts)
        phis = np.linspace(0, 2*np.pi, num=phi_res, endpoint=True)

        # Return distribution matrix, evaluated at phis and ts
        ret_phis = np.zeros((len(ts), phi_res))

        # For every time point, find distribution
        for i, t in enumerate(ts): ##Parallelize?
            sigma_d = np.sqrt(2*self.phase_diffusivity*np.abs(t))
            mu_d = (t * (2*np.pi)/self.period)%(2*np.pi)

            # From the convolution of two gaussian functions
            mu_f = self.mu + mu_d if advance_t else self.mu,

            # Allow negative times to reverse diffusion
            var = self.sigma**2 + np.sign(t)*sigma_d**2
            sigma_f = np.sqrt(var) if var > 0 else 0.

            ret_phis[i] = wrapped_gaussian(phis, mu_f, sigma_f)

        return ret_phis

    def integrate(self, a, b, phi_offset=0):
        """ implement a wrapped error function eventually? """
        return self.fo_phi_interp.integrate(a-phi_offset, b-phi_offset)







def convolve_pdf(f, g, p_diff):
    """ Return the convolution of two periodic probability density
    functions, f and g, with sampling frequency p_diff """
    fft_f = np.fft.fft(f)
    fft_g = np.fft.fft(g)
    return p_diff*np.real(np.fft.ifft(fft_f * fft_g))


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

        def normal_cdf(x):
            return 0.5*(1 + erf((x - mu)/(sigma * sqrt(2))))

        # norm = 1/(sigma*np.sqrt(2*np.pi))

        phi_diff = (2*np.pi)/(len(phis) - 1)
        phis_lower = phis - phi_diff/2
        phis_upper = phis + phi_diff/2

        tsum = np.zeros(len(phis))
        for i in xrange(1000): # Maximum iterations
            oldtsum = np.array(tsum)

            # add both +/- i
            if i == 0: 
                tsum += normal_cdf(phis_upper) - normal_cdf(phis_lower)
            
            else:
                tsum += (normal_cdf(phis_upper + 2*np.pi*i) -
                         normal_cdf(phis_lower + 2*np.pi*i))
                tsum += (normal_cdf(phis_upper - 2*np.pi*i) -
                         normal_cdf(phis_lower - 2*np.pi*i))
            
            diff = (tsum - oldtsum)

            if diff.max() < abstol: break

        return tsum/phi_diff

    mu = np.asarray(mu)%(2*np.pi)

    if sigma > 1.4:
        return wrapped_gaussian_fourier(phis, mu, sigma, abstol)
    else: return wrapped_gaussian_direct(phis, mu, sigma, abstol)

def mean_std(z):
    """ Return the mean angle and circular standard deviation of the
    complex variable z """

    return (np.angle(z), np.sqrt(-2*np.log(np.abs(z))))

def mean_length(z):
    """ Return the mean angle and resultant length of the complex
    variable z """

    return (np.angle(z), np.abs(z))

def normalize(angles, end=np.pi):
    """ normalize angles to the range -Pi, Pi """

    angles = np.atleast_1d(angles)

    angles = angles % (2*np.pi)
    angles[angles > end] += -2*np.pi

    return angles.squeeze()




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

        phi = np.atleast_1d(phi)

        # Must map phis to (0, 2Pi) for interpolation to work correctly
        if np.any(phi > 2*np.pi) or np.any(phi < 0):
            # RectSpline needs arguments in order, but we need to return
            # in the order for which the function was called
            phi = phi%(2*np.pi)
            fwd_sort = phi.argsort()
            rev_sort = fwd_sort.argsort()
            out = self.interp_object(phi[fwd_sort], t)
            return out[:,rev_sort,:].squeeze().T

        # Return interpolated object if all phis in range
        else: return self.interp_object(phi, t).squeeze().T













if __name__ == "__main__":

    # from CommonFiles.Models.degmodelFinal import model, paramset
    from CommonFiles.Models.leloup16model import model, paramset
    # from CommonFiles.Models.tyson2statemodel import model, paramset
    import matplotlib.pylab as plt
    from CommonFiles.PlotOptions import (PlotOptions, format_2pi_axis,
                                         layout_pad)

    PlotOptions(uselatex=True)

    test = Amplitude(model(), paramset)
    # state_pulse_creator = test._s_pulse_creator(1, 0.5)
    param = test.pdict['vsP']
    amount = 0.10*paramset[param]
    duration = np.pi/8
    state_pulse_creator = test._p_pulse_creator(param, amount, duration)
    test.calc_pulse_responses(state_pulse_creator, trans_duration=3)

    # # Fig 1 : Test single cell PRC and ARC
    # fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
    # axmatrix[0].plot(test.phis, test.prc_single_cell)
    # axmatrix[0].set_title('Single Cell PRC')
    # axmatrix[1].plot(test.phis, test.arc_single_cell[:,0])
    # axmatrix[1].set_title('Single Cell ARC')
    # plot_grey_zero(axmatrix[0])
    # plot_grey_zero(axmatrix[1])
    # format_2pi_axis(axmatrix[1])


    period = 2*np.pi
    mean = np.pi/4
    std = 0.5
    decay = 0.01
    phis = np.linspace(0, 2*np.pi, num=100, endpoint=True)
    po = wrapped_gaussian(phis, mean, std)
    po_interp = PeriodicSpline(phis, po)
    
    test_population = gaussian_phase_distribution(mean, std, decay,
                                                  invert_res=60)
    perturbed_popul = test_population.invert(test.phis,
                                             test.prc_single_cell)

    test.calc_population_responses(test_population, tarc=False)

    mean_p, std_p = mean_std(test.z_hat(0))
    mean_pert_pop = gaussian_phase_distribution(mean_p, std_p, decay)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    iphis = np.linspace(0, 2*np.pi, 60)

    for i, color in zip([0., 0.25, 0.5, 0.75], ['r', 'g', 'b', 'y']):
        ax.plot(phis, test_population(i*period).T, color=color,
                label='unperturbed')
        ax.plot(iphis, perturbed_popul(i*period, phi_res=60).T, '--',
                color=color, label='perturbed (exact)')
        ax.plot(phis, mean_pert_pop(i*period).T, ':', color=color,
                label='perturbed (mean+std)')

    format_2pi_axis(ax)

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


    t0 = 0.
    ts_ref = np.linspace(-2*np.pi, test.arc_traj_ts[-1], 250)
    try: ts = test.comb_interp.ts
    except AttributeError: ts = test.traj_interp.ts

    arc_spl = PeriodicSpline(test.phis, test.arc_population, k=4)
    arc_d = arc_spl.derivative()
    roots = arc_d.roots()
    global_min_root = roots[arc_spl(roots).argmin()]
    

    phi_offset = 0

    xbar = test.x_bar(ts_ref, phi_offset=phi_offset)
    xhat = test.x_hat(ts, phi_offset=phi_offset)
    xhat_ss = test.x_hat_ss(ts, phi_offset=phi_offset)
    xhat_ss2 = test.x_hat_ss(ts, phi_offset=phi_offset, approx=True)

    state = 3
    # Fig 3 : Test x(t) functions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts_ref, test.lc_phi(ts_ref+phi_offset)[:,state],
            'k--', label=r'$x^\gamma(t)$')
    ax.plot(ts_ref, xbar[:,state], ':', label=r'$\bar{x}(t)$')
    ax.plot(ts, xhat[:,state], 'r', label=r'$\hat{x}(t)$')
    ax.plot(ts, xhat_ss[:,state], '-',
            label=r'$\hat{x}_{ss}(t)$')
    ax.plot(ts, xhat_ss2[:,state], 'g--',
            label=r'$\hat{x}_{ss2}(t)$')
    # ax.plot(ts, xhat_ss_old[:,0], '-',
    #         label=r'$\hat{x}_{ss old}(t)$')
    # ax.plot(ts, xhat[:,0], 'r--', label=r'$\hat{x}(t)$')
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


    def pop_difference(ts, invert_res):

        test_population.invert_res = invert_res
        perturbed_popul = test_population.invert(test.phis,
                                                 test.prc_single_cell)

        return np.array([np.linalg.norm(perturbed_popul(t) -
                                        mean_pert_pop(t)) for t in ts])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for invert_res in [20, 30, 40, 50, 60, 70, 80]:
    #     ax.semilogy(ts, pop_difference(ts, invert_res),
    #                 label=str(invert_res))
    # ax.set_xlabel('Time (hr)')
    # ax.set_ylabel('Error of mean vs. calculated pdf')
    # ax.legend(loc='best', ncol=2)
    # fig.tight_layout(**layout_pad)




    plt.show()
