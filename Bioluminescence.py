import math

import numpy as np

from scipy import signal, interpolate, optimize, sparse
from scipy.sparse import dia_matrix, eye as speye
from scipy.sparse.linalg import spsolve

# import matplotlib.pylab as plt

from matplotlib import mlab

import pywt


class Bioluminescence(object):
    """ Class to analyze damped bioluminesence data from recordings of
    population level rhythms. Includes algorithms for smoothing,
    detrending, and curve fitting. """

    def __init__(self, x, y):
        """ x, y should contain time and expression level, respectively,
        of a circadian oscillation """

        self.x = x
        self.y = y

        self.xvals = {'raw' : x}
        self.yvals = {'raw' : y}

        self.period = estimate_period(x, y)
        self.even_resample(res=len(x))
    

    def even_resample(self, res=None):
        """ Ensure even data sampling """
        self.x, self.y = even_resample(self.x, self.y, res=res)
        self.xvals['even'] = self.x
        self.yvals['even'] = self.y


    def detrend(self):
        """ Detrend the data """

        self.x, self.y, mean = detrend(self.x, self.y,
                                       est_period=self.period,
                                       ret='both')

        self.yvals['detrended'] = self.y
        self.yvals['mean'] = mean


    def filter(self, cutoff_period=15.):
        """ Lowpass filter to remove noise """
        self.x, self.y = lowpass_filter(self.x, self.y,
                                        cutoff_period=(self.period *
                                                       cutoff_period/24.))
        self.yvals['filt'] = self.y


    def fit_sinusoid(self, t_trans=0.):
        """ Fit a decaying sinusoid, ignoring the part of the signal
        with x less than t_trans """

        start_ind = ((self.x - t_trans)**2).argmin()

        pars, conf = fit_decaying_sinusoid(self.x[start_ind:],
                                           self.y[start_ind:])
        self.sinusoid_pars = pars
        self.sinusoid_confidence = conf
        self.period = pars['period']

        self.yvals['model'] = decaying_sinusoid(self.x,
                                                *_pars_to_plist(pars))

    def amplify_decay(self, amp=None, decay=None):
        """ Function to amplify the tail end of a trace by removing the
        estimated exponential decay. Amplitude and decay parameters, if
        None, will be taken from the estimated sinusoid parameters. """

        if decay is None: decay = self.sinusoid_pars['decay']
        if amp is None: amp = self.sinusoid_pars['amplitude']

        exp_traj = amp*np.exp(-decay * self.x)
        
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

        l = np.all(np.vstack([period_bins[:,0] <= self.period,
                              period_bins[:,1] >= self.period]), axis=0)
        circadian_bin = int(np.where(l)[0]) 

        self.dwt = out

        self.yvals['dwt_detrend'] = out['components'][circadian_bin]
        self.y = out['components'][circadian_bin]

    def reset(self):
        """ reset values in x and y to the raw values used when
        initiating the class """
        self.x = self.xvals['raw']
        self.y = self.yvals['raw']


    def _cwt_y_process(self, x, y):
        """ Process the x and y variables to extrapolate y such that it
        ends at an extrema. Should reduce the boundary effects when
        using mirror end conditions """





    def continuous_wavelet_transform(self, y=None, process_y=True,
                                     shortestperiod=15,
                                     longestperiod=40, nvoice=512,
                                     be=5):
        """ Function to calculate the continuous wavelet transform of
        the data, with an attempt to reduce boundary effects through
        mirroring the data series """

        if y is None:
            try: y = self.yvals['exp_amp']
            except KeyError: y = self.y

        x_len = len(self.x)
        assert len(y) == x_len, "CWT data length mismatch"

        cwt = continuous_wavelet_transform(self.x, y,
                                           shortestperiod=shortestperiod,
                                           longestperiod=longestperiod,
                                           nvoice=nvoice, be=be)

        self.cwt = cwt
        









        




        # Get local extrema
        # max_extrema = np.array(np.diff(np.sign(np.diff(y))), dtype=bool) 
        # max_extrema = np.where(max_extrema)[0]

        # curr_res = len(self.x)
        # dist = curr_res - max_extrema[-1]
        # period = max_extrema[-2] - max_extrema[-1]
        # assert dist < period, "Peak Finding Issue with BioClass.CWT"

        # x_temp = self.x[:max_extrema[-1]]
        # y_temp = y[:max_extrema[-1]]

        # rev_y = y_temp[max_extrema[-2]:max_extrema[-1]][::-1]
        # y_temp = np.hstack([y_temp, rev_y])
        # ext_x = np.hstack([x_temp, x_temp[:period] +
        #                    x_temp[max_extrema[-1]]])


                             








        

            
        
        






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


def lowpass_filter(x, y, cutoff_period=15., order=5):
    """ Filter the data with a lowpass filter, removing noise with a
    critical frequency corresponding to the number of hours specified by
    cutoff_period. Assumes a period of 24h, with data in x in the units
    of hours. """

    x = np.asarray(x)
    y = np.asarray(y)
    nyquist = (x[1] - x[0])/2.
    cutoff_freq = 1/((cutoff_period/(x.max() - x.min()))*len(x))

    b, a = signal.butter(5, cutoff_freq/nyquist)
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

    # As recommended by Ravn, Uhlig 2004, a calculated empiracally 
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
    return periods, signal.lombscargle(x, y, freqs)


def estimate_period(x, y, period_low=1, period_high=100, res=200):
    """ Find the most likely period using a periodogram """
    periods, pgram = periodogram(x, y)
    return periods[pgram.argmax()]


def power_spectrum(x, y):
    """ Return power at each frequency """

    ps = np.abs(np.fft.rfft(y))
    rate = x[1] - x[0]
    freqs = np.linspace(0, rate/2, len(ps))
    
    return freqs, np.log(ps)



def fit_exponential(x, y):
    lny = np.log(y)
    xy = (x * y).sum()
    xxy = (x**2 * y).sum()
    ylny = (y * lny).sum()
    xylny = (x * y * lny).sum()
    y_ = y.sum()

    denom = y_ * xxy - xy**2

    a = (xxy * ylny - xy*xylny)/denom
    b = (y_ * xylny - xy * ylny)/denom

    return np.exp(a), b



def estimate_sinusoid_pars(x, y):
    """ Use a fourier transform based technique to estimate the period,
    phase, and amplitude of the signal in y to prepare for curve fitting
    """

    # Estimate exponential decay
    # Get locations of extrema
    inds = np.where(np.diff(np.sign(np.diff(y))))[0] + 1
    y_extreme = np.abs(y[inds])
    x_extreme = x[inds]

    a, b = fit_exponential(x_extreme, y_extreme)
    

    # Estimate period, phase, and amp using fourier methods
    Y = np.fft.fft(y)
    n = len(Y)
    freqs = np.fft.fftfreq(n, d=x[1] - x[0])
    Y = Y[1:(n/2)]
    freqs = freqs[1:(n/2)]
    ind = np.abs(Y).argmax()
    pars = {
        'period'    : 1/freqs[ind],
        'phase'     : np.angle(Y)[ind],
        # 'amplitude' : np.abs(Y[ind])/(n/2),
        'amplitude' : a,
        'decay'     : b,
    }

    return pars


def decaying_sinusoid(x, amplitude, period, phase, decay):
    """ Function to generate the y values for a fitted decaying sinusoid
    specified by the dictionary 'pars' """
    return (amplitude * np.cos((2*np.pi/period)*x + phase) *
            np.exp(-decay*x))

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

    popt, pcov = optimize.curve_fit(decaying_sinusoid, x, y, p0=p0,
                                    sigma=weights)

    # Find appropriate confidence intervals
    relative_confidence = 2*np.sqrt([pcov[i,i] for i in
                                     xrange(len(pcov))])/np.abs(popt)
    # Normalize amplitude and phase parameters
    if popt[0] < 0:
        popt[0] = abs(popt[0])
        popt[2] += np.pi
    popt[2] = popt[2]%(2*np.pi)

    pars = _plist_to_pars(popt)
    pars_confidence = _plist_to_pars(relative_confidence)

    return pars, pars_confidence


def continuous_wavelet_transform(x, y, shortestperiod=20,
                                 longestperiod=30, nvoice=512, ga=3,
                                 be=7, opt_b='exp_sin', opt_m='ban'):
    """ Call lower level functions to generate the cwt of the data in x
    and y, according to various parameters. Will resample the data to
    ensure efficient fft's if need be """


    lenx = len(x)
    power = np.log(lenx)/np.log(2)

    # Resample the data if lenx is not 2**n
    if bool(power%1): x, y = even_resample(x, y, res=2**int(power))

    fs, tau, qscaleArray = calculate_widths(x, shortestperiod,
                                            longestperiod, nvoice)

    wt = cwt(y, fs, ga, be, opt_b, opt_m)

    cwt_abs = np.abs(wt)
    cwt_scale = cwt_abs/cwt_abs.sum(0)
    cwt_angle = np.angle(wt)

    max_inds = cwt_abs.argmax(0)
    period = tau[max_inds]

    wt_max_inds = np.ravel_multi_index((max_inds, np.arange(len(x))),
                                       wt.shape)

    phase = cwt_angle.flat[wt_max_inds]
    amplitude = cwt_abs.flat[wt_max_inds]
    
    return_dict = {
        'x'         : x,
        'tau'       : tau,
        'cwt'       : wt,
        'cwt_abs'   : cwt_abs,
        'cwt_scale' : cwt_scale,
        'cwt_angle' : cwt_angle,
        'period'    : period,
        'phase'     : phase,
        'amplitude' : amplitude,
        'max_inds'  : wt_max_inds,
    }

    return return_dict


def calculate_widths(x, shortestperiod=20., longestperiod=30.,
                     nvoice=512):
    """ Adaptation of Tania Leise's code from
    http://www3.amherst.edu/~tleise/CircadianWaveletAnalysis.html """

    T = x[-1] - x[0] 
    NstepsPerHr = len(x)/T

    scale = np.floor(2*np.pi*T/longestperiod)
    noctave = int(np.ceil(np.log2(2*np.pi*T/(scale*shortestperiod))))

    scale_arr = np.hstack([np.ones(nvoice)*scale*(2**i) for i in
                           xrange(noctave)])

    pow_arr = np.hstack([np.arange(nvoice) for i in xrange(noctave)])+1

    qscaleArray = (scale_arr * 2**(pow_arr/float(nvoice)))[::-1]

    tau=(2*np.pi)*T/qscaleArray
    fs=2*np.pi/(tau*NstepsPerHr)

    valid_inds = np.all(np.vstack([tau >= shortestperiod,
                                   tau <= longestperiod]), 0)

    return fs[valid_inds], tau[valid_inds], qscaleArray[valid_inds]


def cwt(y, fs, ga=3, be=7, opt_b='exp_sin', opt_m='ban'):
    """
    Calculate the continuous wavelet transform using generalized morse
    wavelets. 

    Parameters
    ----------
    y      : data signal
    fs     : frequency bins
    ga     : gamma, parameter
    be     : beta, parameter
    opt_b  : 'zer', 'mir', or 'per'; determines method of handling edge
           : effects. Zero-padding, Mirroring, or assuming the signal is
           : periodic.
    opt_m  : 'ene' or 'ban', which determines the type of normalization
           : to use.
           
    Returns
    -------
    cwt    : Continuous wavelet transform
    """

    bdetrend = 1
    # fam = 'first'
    # dom = 'frequency'

    x, index = timeseries_boundary(y, opt_b, bdetrend)

    M = len(x)
    # N = len(fs)

    # X = np.zeros((M, N))

    fo = np.exp((1./ga)*(np.log(be) - np.log(ga)))
    fact = 2*np.pi*fo/fs
    om = fact*np.atleast_2d(np.linspace(0, 1, M, endpoint=False)).T

    # Matricies are very sparse, use sparse matricies to speed the
    # calculations 

    # 1/e^(om**ga) will zero out anything when om[i,j] > 709 (maximum
    # value for a double precision float)
    sparsity_inds = om >= (709.79)**(1./ga)

    sparsity_inds[int(round(M/2)):] = True

    om[sparsity_inds] = 0

    om_s = sparse.csr_matrix(om)
    om_ga = om_s.copy()
    om_be = om_s.copy()

    om_ga.data = om_s.data**ga
    om_be.data = om_s.data**be

    # psizero = om_be/np.exp(om_ga) 
    # Need to do some manipulation here
    om_ga.data = np.exp(om_ga.data)
    om_ga.data = 1/om_ga.data

    psizero = om_be.multiply(om_ga)

    if opt_m is 'ban':
        psizero.data *= 2*fo**(-be)*np.exp(fo**ga)
        
    psizero[0].data *= 0.5

    # psizero[np.isnan(psizero)] = 0

    r = (2*be + 1)/ga
    
    coeff = 1

    if opt_m is 'ene':
        A = np.sqrt((np.pi*ga*(2**r)*np.exp(math.lgamma(1) -
                                            math.lgamma(r))))
        coeff = np.sqrt(2./fact)*A

    elif opt_m is 'ban':
        coeff = np.sqrt(math.gamma(1))

    W = psizero
    W.data *= coeff
    # W[np.isinf(W)] = 0 # Not sure if this is needed
    W = W.conj()

    # Find frequency domain of input data, convert to sparse matrix for
    # multiplication
    X = np.fft.fft(x)
    X = sparse.csr_matrix(np.atleast_2d(X).T)

    # Convolution of wavelets and input data
    iT = W.multiply(X)


    T = np.fft.ifft(iT.todense(), axis=0)[index, :]
    
    return T.T

def timeseries_boundary(x, opt_b, bdetrend):
    """ TIMESERIES_BOUNDARY applies periodic, zero-padded, or mirror
    boundary conditions to a time series. """

    M = len(x)

    if bool(M%2): raise AssertionError("Even number of samples")
     
    if bdetrend: x = mlab.detrend_linear(x)

    # Allocate space for solution
    if opt_b == "zer":
        y = np.hstack([np.zeros(M/2), x, np.zeros(M/2)])
    elif opt_b == "mir":
        y = np.hstack([x[::-1][M/2:], x, x[::-1][:M/2]])
    elif opt_b == "per":
        y = x
    
    elif opt_b == "exp_sin":
        # Attempt to fit an exponential periodic function to each end of
        # the signal, such that the total length is 2*M

        # Declare a temporary variable t, all my functions operate on an
        # x and y pair rather than dimensionless frequency
        t = np.linspace(0, 100, len(x))
        period = estimate_period(t, x)
        
        # Get last two periods of x and y
        ind_end   = np.abs(t - (t[-1] - 2*period)).argmin()
        ind_start = np.abs(t - (2*period)).argmin()
        t_end   = t[ind_end:]
        x_end   = x[ind_end:]
        t_start = t[:ind_start]
        x_start = x[:ind_start]
        x_mid   = x[ind_start:ind_end]

        t_end_ext, x_end_ext     = extend(t_end, x_end, M/2)
        t_start_ext, x_start_ext = extend(t_start, x_start[::-1], M/2)

        y = np.hstack([x_start_ext[::-1], x_mid, x_end_ext])
        # tt = np.linspace(t[0] - t.mean(), t[-1] + t.mean(), num=len(yt))

    if opt_b is not 'per': index = np.arange(M/2, M + M/2)
    else: index = np.arange(M)

    return y, index

def extend(x_end, y_end, length):
    """ Extends x_end, y_end by length, using a decaying sinusoid. Will
    only work for detrended data. Tries to respect original data in
    y_end, but some may be changed to keep the function smooth. Assumes
    x_end, y_end encompass approximately 2 full periods"""

    end_len = len(x_end)
    period = (x_end[-1] - x_end[0])/2
    dx = (x_end[-1] - x_end[0])/(end_len - 1)
    extnum = length + end_len
    x_ext = np.linspace(x_end[0], x_end[0] + extnum*dx, num=extnum,
                        endpoint=False)

    weights = np.exp(-3*(x_end-x_end[0])/period)

    end_pars, end_pcov = fit_decaying_sinusoid(x_end, y_end,
                                               weights=weights)

    y_fit = decaying_sinusoid(x_ext, *_pars_to_plist(end_pars))

    def hill(t, K, start=0, finish=1, n=3):
        return start + (finish-start)*(t**n/(K**n + t**n))

    # Smooth transition from y_end to y_fit
    merge = hill(x_end, x_end[-1] - period/2, n=100)

    y_ext = np.zeros(*x_ext.shape)
    y_ext[:end_len] += y_end * (1 - merge)
    y_ext[:end_len] += y_fit[:end_len] * (merge)
    y_ext[end_len:] = y_fit[end_len:]

    return x_ext, y_ext



def timeseries_boundary_old(x, opt_b, bdetrend):
    """ TIMESERIES_BOUNDARY applies periodic, zero-padded, or mirror
    boundary conditions to a time series. """

    M = len(x)
    
    if bdetrend: x = mlab.detrend_linear(x)

    # Allocate space for solution
    if opt_b == "zer":
        y = np.hstack([np.zeros(M), x, np.zeros(M)])
    elif opt_b == "mir":
        y = np.hstack([x[::-1], x, x[::-1]])
    elif opt_b == "per":
        y = x

    if opt_b is not 'per': index = np.arange(M, 2*M)
    else: index = np.arange(M)

    return y, index

def calculate_widths_old(x, shortestperiod=20., longestperiod=30.,
                     nvoice=512):
    """ Adaptation of Tania Leise's code from
    http://www3.amherst.edu/~tleise/CircadianWaveletAnalysis.html """

    T = x[-1] - x[0] 
    NstepsPerHr = len(x)/T

    scale = np.floor(2*np.pi*T/longestperiod)
    noctave = np.ceil(np.log2(2*np.pi*T/(scale*shortestperiod)))
    nscale  = nvoice * noctave
    kscale  = 1

    qscaleArray = np.zeros(nscale)
    for jo in xrange(1, int(noctave)+1):
        for jv in xrange(1, int(nvoice)+1):
            qscale = scale * (2**(float(jv)/nvoice))
            qscaleArray[nscale-kscale] = float(qscale)
            kscale = kscale+1
        scale = scale * 2

    tau=(2*np.pi)*T/qscaleArray
    fs=2*np.pi/(tau*NstepsPerHr)

    return fs, tau, qscaleArray



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

    return {
        'period_bins' : period_bins,
        'components' : rec_d,
    }


    


    




    

# if __name__ == "__main__":
#     import matplotlib.pylab as plt
#     from CommonFiles.PlotOptions import (PlotOptions, plot_grey_zero,
#                                          layout_pad)
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
