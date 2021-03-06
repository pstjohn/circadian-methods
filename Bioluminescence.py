import math

import numpy as np

from scipy import signal, interpolate, optimize, sparse
from scipy.sparse import dia_matrix, eye as speye
from scipy.sparse.linalg import spsolve

# import matplotlib.pylab as plt

from matplotlib import mlab
from CommonFiles.Utilities import (color_range, format_number,
                                   round_sig)

import pywt
import pycwt


class Bioluminescence(object):
    """ Class to analyze damped bioluminesence data from recordings of
    population level rhythms. Includes algorithms for smoothing,
    detrending, and curve fitting. """

    def __init__(self, x, y, period_guess=None):
        """ x, y should contain time and expression level, respectively,
        of a circadian oscillation """

        self.x = x
        self.y = y

        self.xvals = {'raw' : x}
        self.yvals = {'raw' : y}

        if not period_guess:
            period_low  = period_guess/2. if period_guess else 1
            period_high = period_guess*2. if period_guess else 100
            self.period = estimate_period(x, y, period_low=period_low,
                                          period_high=period_high)
        else:
            # Until scipy fixes their periodogram issues
            self.period = period_guess

        self.even_resample(res=len(x))
    

    def even_resample(self, res=None):
        """ Ensure even data sampling """
        self.x, self.y = even_resample(self.x, self.y, res=res)
        self.xvals['even'] = self.x
        self.yvals['even'] = self.y


    def _exp_detrend(self):
        """ Some bioluminescence profiles have mean dynamics
        well-described by an exponentially decaying sinusoid. """

        a,d = fit_exponential(self.x, self.y)
        mean = a*np.exp(self.x*d)
        self.y = self.y - mean

        self.yvals['detrended'] = self.y
        self.yvals['mean'] = mean


    def detrend(self, a=0.05, detrend_period=None):
        """ Detrend the data """

        if detrend_period is None: detrend_period = self.period

        self.x, self.y, mean = detrend(self.x, self.y,
                                       est_period=detrend_period,
                                       ret='both', a=a)

        self.yvals['detrended'] = self.y
        self.yvals['mean'] = mean


    def filter(self, cutoff_period=15.):
        """ Lowpass filter to remove noise """
        self.x, self.y = lowpass_filter(self.x, self.y,
                                        cutoff_period=(self.period *
                                                       cutoff_period/24.))
        self.yvals['filt'] = self.y


    def estimate_sinusoid_pars(self, t_trans=0.):
        """ Estimate decaying sinusoid parameters without fitting """

        self.start_ind = start_ind = self._ind_from_x(t_trans)
        return estimate_sinusoid_pars(self.x[start_ind:],
                                      self.y[start_ind:])

    def _ind_from_x(self, x):
        return ((self.x - x)**2).argmin()

    def fit_sinusoid(self, t_trans=0., weights=None):
        """ Fit a decaying sinusoid, ignoring the part of the signal
        with x less than t_trans """

        self.start_ind = start_ind = ((self.x - t_trans)**2).argmin()

        pars, conf = fit_decaying_sinusoid(self.x[start_ind:],
                                           self.y[start_ind:],
                                           weights=weights)
        self.sinusoid_pars = pars
        self.sinusoid_confidence = conf
        self.period = pars['period']

        self.yvals['model'] = decaying_sinusoid(self.x,
                                                *_pars_to_plist(pars))

    def pseudo_r2(self):
        """Calculates the pseudo-r2 value for the fitted sinusoid."""
        y_reg = self.yvals['model'][self.start_ind:]
        y_dat = self.y[self.start_ind:]
            
        SSres = ((y_dat - y_reg)**2).sum()
        SStot = ((y_dat - y_dat.mean())**2).sum()
        return 1 - SSres/SStot

    def amplify_decay(self, amp=None, decay=None):
        """ Function to amplify the tail end of a trace by removing the
        estimated exponential decay. Amplitude and decay parameters, if
        None, will be taken from the estimated sinusoid parameters. """

        if decay is None: decay = self.sinusoid_pars['decay']
        if amp is None: amp = self.sinusoid_pars['amplitude']

        # Decay in units of radians
        exp_traj = amp*np.exp(-decay * 2*np.pi*self.x/self.period)
        
        # Normalize by first value
        # exp_traj *= 1./exp_traj[0]

        self.yvals['exp_amp'] = self.y/exp_traj


    def dwt_breakdown(self, best_res=None, wavelet='dmey', mode='sym'):
        """ Break the signal down into component frequencies using the
        dwt. 

        - This function should select an optimum sampling frequency to
          let the circadian range fall within the middle of a frequency
          bin 

        - Also detrends the signal
          
          """

        # Sample the interval with a number of samples = 2**n
        

        if best_res is None:
            curr_res = len(self.x)
            curr_pow = int(np.log(curr_res)/np.log(2))

            def bins(curr_res):
                curr_pow = int(np.log(curr_res)/np.log(2))
                dx = (self.x[-1] - self.x[0])/(curr_res - 1)
                period_bins = np.array([(2**j*dx, 2**(j+1)*dx) for j in
                                        xrange(1,curr_pow+1)])
                l = np.all(np.vstack([period_bins[:,0] <= self.period,
                                      period_bins[:,1] >= self.period]),
                           axis=0)
                circadian_bin = int(np.where(l)[0]) 
                return np.abs(np.sum((period_bins[circadian_bin] -
                                      self.period)))

            best_res = optimize.fminbound(bins, 2**(curr_pow-1),
                                          2**(curr_pow+1))

        self.even_resample(res=int(best_res))

        # self.even_resample(res=2**(curr_pow))

        out = dwt_breakdown(self.x, self.y, wavelet=wavelet,
                            nbins=np.inf, mode=mode)

        period_bins = np.array(out['period_bins'])
        self.dwt_bins = len(period_bins)

        l = np.all(np.vstack([period_bins[:,0] <= self.period,
                              period_bins[:,1] >= self.period]), axis=0)
        circadian_bin = int(np.where(l)[0]) 

        self.dwt = out

        self.yvals['dwt_detrend'] = out['components'][circadian_bin]
        self.dwt['circadian_bin'] = circadian_bin
        self.y = out['components'][circadian_bin]


    def continuous_wavelet_transform(self, y=None, dj=None, NJ=7,
                                           w_morlet=4):
        """ Function to calculate the continuous wavelet transform of
        the data, with an attempt to reduce boundary effects through
        mirroring the data series """

        if y is None: y = self.y

        x_len = len(self.x)
        assert len(y) == x_len, "CWT data length mismatch"

        if dj is None:
            # Match vertical to horizontal resolution
            dj = NJ/float(x_len)

        cwt = continuous_wavelet_transform(self.x, y, dj=dj, NJ=NJ,
                                           w_morlet=w_morlet)

        self.cwt = cwt

    def plot_cwt(self, ax):



        assert hasattr(self, 'cwt'), "Must perform CWT first"

        # Expand return dictionary from cwt
        power = self.cwt['cwt_power']
        coi   = self.cwt['coi']
        x     = self.cwt['x']
        y     = self.cwt['tau']
        sig95 = self.cwt['sig95']
        ridge = self.cwt['ridge']
        dt    = self.cwt['dt']

        # Normalize power
        # power -= power.mean()
        power /= power.std()

        # levels = [2**i for i in xrange(-4, 5)]
        levels = power.mean() + np.array([1, 2, 5, 10])

        X, Y = np.meshgrid(x, y)

        # Heat map of CWT power
        from matplotlib.colors import LogNorm

        ax.contour(x, y, power, levels=levels, colors='k', linewidths=0.5)
        ax.contourf(x, y, power, levels=levels, cmap='PuBu',
                    norm=LogNorm())
        # ax.pcolormesh(X, Y, power, cmap='PuBu', norm=LogNorm())

        # Significance contours
        ax.contour(x, y, sig95, [-99, 1], colors='k',
                   linewidths=1.)

        # Wavelet ridge
        ax.plot(x, y[ridge], 'r')

        # Shade COI region
        ax.fill(
            np.concatenate([x, x[-1:] + dt, x[-1:] + dt, x[:1] - dt,
                            x[:1] - dt]),
            np.concatenate([coi,[1e-9], y[-1:], y[-1:], [1e-9]]),
            'k', alpha=0.3, hatch='x')

        # Set up tick locations
        Yticks = 2 ** np.arange(np.ceil(np.log2(y.min())),
                                   np.ceil(np.log2(y.max())))
        ax.set_yscale('log', basey=2)
        ax.set_ylim([y.min(), y.max()])
        ax.set_yticks(Yticks)
        ax.set_yticklabels(Yticks)
        ax.set_xlim([x.min(), x.max()])
        ax.invert_yaxis()




        

    def reset(self):
        """ reset values in x and y to the raw values used when
        initiating the class """
        self.x = self.xvals['raw']
        self.y = self.yvals['raw']



    def plot_dwt_components(self, ax, space=1.0, bins=None,
                            baselines=None):
        """ Plot the decomposition from the dwt on the same set of axes,
        similar to figure 4 in DOI:10.1177/0748730411416330. space is
        relative spacing to leave between each component, bins is a
        boolean numpy array which specifies which components to plot
        (default is all bins) """

        if bins is None:
            bins = np.array([True]*self.dwt_bins)
        components = np.array(self.dwt['components'])[bins]

        if baselines is None:
            # Assume that dwt breakdown will be part of a shared x/y
            # subplot

            baselines = [0,]
            last_component = np.zeros(components[0].shape)

            spacing = space*(((components.max(1) -
                             components.min(1)).sum())/self.dwt_bins)
            
            for c in components:
                width = np.abs((c-last_component).min()) + spacing
                baselines += [width + baselines[-1]]
                last_component = c

            baselines = np.array(baselines[1:]) 
            self.dwt['plot_baselines'] = baselines

            periods    = np.array(self.dwt['period_bins'])
            period_str = format_number(round_sig(periods))

            ax.set_yticks(baselines)
            ax.set_xlim([self.x.min(), self.x.max()])
            ax.set_ylim([0, baselines[-1] + components[-1].max() +
                         spacing])
            ax.set_yticklabels([pr[0] + 'h - ' + pr[1] +'h' for pr in
                                period_str])


        components = components + np.atleast_2d(baselines).T
        for comp, color in zip(components, color_range(self.dwt_bins)):
            ax.plot(self.x, comp, color=color)

        return ax

    def hilbert_envelope(self, y=None):
        """ Calculate the envelope of the function (amplitude vs time)
        using the analytical signal generated through a hilbert
        transform """

        if y is None: y = self.y
        return abs(signal.hilbert(y))

    
    def fit_hilbert_envelope(self, t_start=None, t_end=None):
        """ Fit an exponential function to the hilbert envelope. Due to
        ringing at the edges, it is a good idea to cut some time from
        the start and end of the sinusoid (defaults to half a period at
        each edge) """

        if t_start is None: t_start = self.period/2
        if t_end is None: t_end = self.period/2

        start = self._ind_from_x(t_start)
        end = self._ind_from_x(t_end)

        envelope = self.hilbert_envelope()[start:end]
        amplitude, decay = fit_exponential(self.x[start:end], envelope)

        return amplitude, -decay*self.period/(2*np.pi) # (1/rad)



def bandpass_filter(x, y, low=10, high=40., order=5):
    """ Filter the data with a lowpass filter, removing noise with a
    critical frequency corresponding to the number of hours specified by
    cutoff_period. Assumes a period of 24h, with data in x in the units
    of hours. """

    x = np.asarray(x)
    y = np.asarray(y) - y.mean()

    nyquist = (x[1] - x[0])/2.

    low_freq = 1/((low/(x.max() - x.min()))*len(x))
    high_freq = 1/((high/(x.max() - x.min()))*len(x))
    b, a = signal.butter(5, (high_freq/nyquist, low_freq/nyquist))
    y_filt = signal.filtfilt(b, a, y)

    return x, y_filt


def lowpass_filter(x, y, cutoff_period=5., order=5):
    """ Filter the data with a lowpass filter, removing noise with a
    critical frequency corresponding to the number of hours specified by
    cutoff_period. Assumes a period of 24h, with data in x in the units
    of hours. """

    x = np.asarray(x)
    y = np.asarray(y)
    nyquist = (x[1] - x[0])/2.
    cutoff_freq = 1/((cutoff_period/(x.max() - x.min()))*len(x))

    b, a = signal.butter(order, cutoff_freq/nyquist)
    y_filt = signal.filtfilt(b, a, y)

    return x, y_filt


def even_resample(x, y, res=None, s=None, meth='linear'):
    """ Function to resample the x,y dataset to ensure evenly sampled
    data. Uses an interpolating spline, with the default resolution set
    to the current length of the x vector. """

    x = np.asarray(x)
    y = np.asarray(y)
    assert len(x) == len(y), "Resample: Length Mismatch"
    if res == None: res = len(x)

    x_even = np.linspace(x.min(), x.max(), res)

    if meth == 'linear':
        interp_func = interpolate.interp1d(x, y, kind='linear')
        y_even = interp_func(x_even)

    if meth == 'spline':
        if s == None: s = 1E-5*len(x)
        spline = interpolate.UnivariateSpline(x, y, s=s)
        y_even = spline(x_even)

    return x_even, y_even


def detrend(x, y, est_period=24., ret="detrended", a=0.05):
    """ Detrend the data using a hodrick-prescott filter. If ret ==
    "mean", return the detrended mean of the oscillation. Estimated
    period and 'a' parameter are responsible for determining the optimum
    smoothing parameter """ 

    x = np.asarray(x)
    y = np.asarray(y)

    # yt, index = timeseries_boundary(y, opt_b='mir', bdetrend=False)

    # As recommended by Ravn, Uhlig 2004, a calculated empirically 
    num_periods = (x.max() - x.min())/est_period
    points_per_period = len(x)/num_periods
    w = a*points_per_period**4
    

    y_mean = hpfilter(y, w)
    y_detrended = y - y_mean

    if ret == "detrended": return x, y_detrended
    elif ret == "mean": return x, y_mean
    elif ret == "both": return x, y_detrended, y_mean


def hpfilter(X, lamb):
    """ Code to implement a Hodrick-Prescott with smoothing parameter
    lambda. Code taken from statsmodels python package (easier than
    importing/installing, https://github.com/statsmodels/statsmodels """

    X = np.asarray(X, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)
    I = speye(nobs,nobs)
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = dia_matrix((data, offsets), shape=(nobs-2,nobs))

    trend = spsolve(I+lamb*K.T.dot(K), X, use_umfpack=True)
    return trend




def periodogram(x, y, period_low=1, period_high=35, res=200):
    """ calculate the periodogram at the specified frequencies, return
    periods, pgram """
    
    periods = np.linspace(period_low, period_high, res)
    # periods = np.logspace(np.log10(period_low), np.log10(period_high),
    #                       res)
    freqs = 2*np.pi/periods
    try: pgram = signal.lombscargle(x, y, freqs)
    # Scipy bug, will be fixed in 1.5.0
    except ZeroDivisionError: pgram = signal.lombscargle(x+1, y, freqs)
    return periods, pgram


def estimate_period(x, y, period_low=1, period_high=100, res=200):
    """ Find the most likely period using a periodogram """
    periods, pgram = periodogram(x, y, period_low=period_low,
                                 period_high=period_high, res=res)
    return periods[pgram.argmax()]


def power_spectrum(x, y):
    """ Return power at each frequency """

    ps = np.abs(np.fft.rfft(y))
    rate = x[1] - x[0]
    freqs = np.linspace(0, rate/2, len(ps))
    
    return freqs, np.log(ps)



def fit_exponential(x, y):
    if fit_exponential.weights == 'equal':
        lny = np.log(y)
        xy = (x * y).sum()
        xxy = (x**2 * y).sum()
        ylny = (y * lny).sum()
        xylny = (x * y * lny).sum()
        y_ = y.sum()

        denom = y_ * xxy - xy**2

        a = (xxy * ylny - xy*xylny)/denom
        b = (y_ * xylny - xy * ylny)/denom
    
    else:
        A = np.vstack([x, np.ones(len(x))]).T
        b, a = np.linalg.lstsq(A, np.log(y))[0]

    return np.exp(a), b

fit_exponential.weights = 'equal'


def estimate_sinusoid_pars(x, y):
    """ Use a fourier transform based technique to estimate the period,
    phase, and amplitude of the signal in y to prepare for curve fitting
    """

    hilbert = signal.hilbert(y)

    # Estimate exponential decay
    envelope = abs(hilbert)
    amplitude, decay = fit_exponential(x, envelope)

       
    # Fit line to phase vs time:
    phases = np.unwrap(np.angle(hilbert))
    weights = envelope*tukeywin(len(x), 0.5) # Prioritize high-amplitude
    slope, intercept = np.polyfit(x, phases, 1, w=weights)
    period = (2*np.pi/slope)
    phase = intercept%(2*np.pi)

    # # Estimate period, phase, and amp using fourier methods
    # Y = np.fft.fft(y)
    # n = len(Y)
    # freqs = np.fft.fftfreq(n, d=x[1] - x[0])
    # Y = Y[1:(n/2)]
    # freqs = freqs[1:(n/2)]
    # ind = np.abs(Y).argmax()
    
    pars = {
        'period'    : period,
        'phase'     : phase,
        'amplitude' : amplitude,
        'decay'     : -decay*period/(2*np.pi), # (1/rad)
        # 'weights'   : weights,
    }

    return pars


def decaying_sinusoid(x, amplitude, period, phase, decay):
    """ Function to generate the y values for a fitted decaying sinusoid
    specified by the dictionary 'pars' """
    return (amplitude * np.cos((2*np.pi/period)*x + phase) *
            np.exp(2*np.pi*decay*x/period))

def _pars_to_plist(pars):
    pars_list = ['amplitude', 'period', 'phase', 'decay']
    return [pars[par] for par in pars_list]

def _plist_to_pars(plist):
    pars_list = ['amplitude', 'period', 'phase', 'decay']
    pars = {}
    for par, label in zip(plist, pars_list): pars[label] = par
    return pars

def fit_decaying_sinusoid(x, y, weights=None):
    """ Estimate and fit parameters for a decaying sinusoid to fit y(x)
    """

    p0 = _pars_to_plist(estimate_sinusoid_pars(x, y))
    if weights is None: weights = 1/np.exp(-p0[-1]*x)
    if weights is 'capped': weights = 0.1 + 1/np.exp(-p0[-1]*x)

    popt, pcov = optimize.curve_fit(decaying_sinusoid, x, y, p0=p0,
                                    sigma=weights, maxfev=5000)

    # Find appropriate confidence intervals
    relative_confidence = 2*np.sqrt([pcov[i,i] for i in
                                     xrange(len(pcov))])/np.abs(popt)
    # Normalize amplitude and phase parameters
    if popt[0] < 0:
        popt[0] = abs(popt[0])
        popt[2] += np.pi
    popt[2] = popt[2]%(2*np.pi)

    if popt[0] < 0:
        popt[0] = abs(popt[0])
        popt[2] += np.pi
    popt[2] = popt[2]%(2*np.pi)
    pars_confidence = _plist_to_pars(relative_confidence)
    pars = _plist_to_pars(popt)

    return pars, pars_confidence


def continuous_wavelet_transform(x, y, dj=0.01, NJ=7, w_morlet=4):
    """continuous_wavelet_transform computes the cwt of the input x and
    y data, returning a dictionary of output parameters.

    Parameters
    ----------

    x : 1-D sequence of floats
       The x-coordinates of the data
    y : 1-D sequence of floats
       The y-coordinates of the data
    dj : float
       1/N, where N is the number of sampling points per octave.
       Increases the resulting resolution of the cwt.
    NJ : int
       NJ is the number of octaves to compute.
    w_morlet : int
       w_morlet is the degree of the mother morlet wavelet.

    Returns
    -------
    cwt : dict
        A dictionary of output values.
    """


    # Normalize y
    # std = y.std()                   # Standard deviation
    # var = (y - y.mean()) / std      # Normalizing
    var = y
    
    N = var.size                    # Number of measurements
    dt = x[1] - x[0]                # Sampling frequency
    s0 = -1                         # Use minimum possible period
    J = NJ / dj                     # Maximum period (2**NJ)

    # Calculate red-noise for current spectrum.
    # If errors, use white noise
    try: alpha, _, _ = pycwt.wavelet.ar1(var)       
    except Warning: alpha = 0.0     # Lag-1 autocorr for white noise


    # Main difference from Tania's method - using Morlet wavelets
    mother = pycwt.wavelet.Morlet(w_morlet)  

    # Perform main cwt transform
    wave, scales, freqs, coi, fft, fftfreqs = pycwt.wavelet.cwt(
        var, dt, dj, s0, J, mother)

    power = (abs(wave)) ** 2          # Normalized wavelet power spectrum
    period = 1. / freqs

    slevel = 0.95                     # Significance level

    # Calculate cwt significance from red noise spectrum
    signif, fft_theor = pycwt.wavelet.significance(
        1.0, dt, scales, 0, alpha, significance_level=slevel,
        wavelet=mother)

    # Where ratio > 1, power is significant
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95             

    # Calculate ridge
    ridge = pycwt.proportional_ridge(wave, coi, freqs, max_dist=0.01)

    ridge_inds = np.ravel_multi_index(
        (ridge, np.arange(len(x))), wave.shape)

    angle = np.angle(wave)
    ridge_phase = angle.flat[ridge_inds]
    ridge_amplitude = power.flat[ridge_inds]
    ridge_period = period[ridge]


    
    return_dict = {
        'x'               : x,
        'tau'             : period,
        'cwt'             : wave,
        'cwt_power'       : power,
        'cwt_phase'       : angle,
        'ridge_period'    : ridge_period,
        'ridge_phase'     : ridge_phase,
        'ridge_amplitude' : ridge_amplitude,
        'ridge'           : ridge,
        'ridge_flat'      : ridge_inds,
        'coi'             : coi,
        'sig95'           : sig95,
        'dt'              : dt,
        'ridge_coi'       : ridge_period < coi,
    }

    return return_dict




def dwt_breakdown(x, y, wavelet='dmey', nbins=np.inf, mode='sym'):
    """ Function to break down the data in y into multiple frequency
    components using the discrete wavelet transform """

    lenx = len(x)

    # Restrict to the maximum allowable number of bins
    if lenx < 2**nbins: nbins = int(np.floor(np.log(len(x))/np.log(2)))

    dx = x[1] - x[0]
    period_bins = [(2**j*dx, 2**(j+1)*dx) for j in xrange(1,nbins+1)]

    details = pywt.wavedec(y, wavelet, mode, level=nbins)
    cA = details[0]
    cD = details[1:][::-1]

    # Recover the individual components from the wavelet details
    rec_d = []
    for i, coeff in enumerate(cD):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, wavelet)[:lenx])

    rec_a = pywt.waverec([cA] + [None]*len(cD), wavelet)[:lenx]

    return {
        'period_bins' : period_bins,
        'components' : rec_d,
        'approximation' : rec_a,
    }



def fit_limitcycle_sinusoid(y):
    """ Fit a sinusoid to an input periodic limit cycle, returning the
    amplitude and phase parameters """

    y = np.array(y)
    y += -y.mean()
    ts = np.linspace(0, 2*np.pi, len(y))
    amp_guess = y.std()
    phase_guess = ts[y.argmax()]

    def sin_model(ts, amp, phase):
        return amp*np.sin(ts + phase)

    popt, pcov = optimize.curve_fit(sin_model, ts, y,
                                    p0=[amp_guess, phase_guess])

    return popt


def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be
    regarded as a cosine lobe of width \alpha * N / 2 that is convolved
    with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it
    becomes rectangular, and at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in
    case users compare a MATLAB output to this function output
 
    Reference
    ---------
 
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/
    tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha *
                                           (x[first_condition] -
                                            alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha *
                                           (x[third_condition] - 1 +
                                            alpha/2))) 
 
    return w
    


    


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('../pycwt/sample/circ_hard.csv')
    bclass = Bioluminescence(data.x, data.y, period_guess=24.)

    bclass.detrend()
    bclass.continuous_wavelet_transform()

    import matplotlib.pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bclass.plot_cwt(ax)

    plt.show()

# 
#     PlotOptions()
# 
#     from MelanopsinData import xn, yn
# 
#     est_period = estimate_period(xn, yn)
# 
#     x, y_even = even_resample(xn, yn, res=300)
# 
#     # Add additional noise
#     baseline = 0.5*(1 + np.sin(x*2*np.pi/160))
#     y_even += baseline
#     y_even += 0.1*np.random.rand(y_even.size)
# 
#     x, y_filt = lowpass_filter(x, y_even)
#     x, y_detrend = detrend(x, y_filt, est_period=est_period)
#     x, y_mean = detrend(x, y_filt, est_period=est_period, ret="mean")
# 
#     ind_start = y_detrend.argmin()
#     pars, pars_c = fit_decaying_sinusoid(x[ind_start:],
#                                          y_detrend[ind_start:])
# 
#     x_fit = x[ind_start:]
#     y_fit = decaying_sinusoid(x_fit, *_pars_to_plist(pars))
# 
# 
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
# 
#     ax.plot(x, y_even, '.', label='even_sampling')
#     ax.plot(x, y_filt, label='lowpass_filt')
#     ax.plot(x, y_mean, '--',label='detrended_mean')
#     ax.plot(x, y_detrend, label='detrended')
#     ax.plot(x_fit, y_fit, '--', label='decaying_fit')
#     plot_grey_zero(ax)
#     ax.legend(loc='best')
#     ax.set_xlabel('Time (hr)')
# 
#     fig.tight_layout(**layout_pad)
# 
# 
# 
# 
# 
#     ## Code to calculate a for HP filter (approx 0.05)
#     # def res_sample(res):
#     #     x_even, y_even = even_resample(xn, yn, res=res)
#     #     baseline = 0.5*(1 + np.sin(x_even*2*np.pi/160))
#     #     y_even += baseline
#     #     y_even += 0.1*np.random.rand(y_even.size)
# 
#     #     x, y_filt = lowpass_filter(x_even, y_even)
# 
#     #     def test_detrend(w):
#     #         x_mean, y_mean = detrend(x, y_filt, w=w, ret="mean")
#     #         return np.linalg.norm(y_mean - baseline)
# 
#     #     return optimize.fmin(test_detrend, 1600, disp=False)[0]
# 
#     # res_s = np.linspace(100, 500, 10)
#     # best_w = np.array([res_sample(res) for res in res_s])
# 
# 
#     # def fourth_power(a):
#     #     func = lambda x: a*(x/8.)**4
#     #     return np.linalg.norm(best_w - func(res_s))
# 
#     # import matplotlib.pylab as plt
#     # plt.plot(res_s, best_w)
# 
#     plt.show()
# 
# 
# 
# 
# 
# 
#         
# 
# 
# 
#     
# 
# 
# 
