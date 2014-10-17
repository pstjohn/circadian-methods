import numpy as np

from CommonFiles.Bioluminescence import Bioluminescence

from lmfit import minimize, Parameters
# from statsmodels.sandbox.stats.multicomp import multipletests
# from scipy import stats

class SingleModel(object):

    def __init__(self, x, y, degree):
        self.x = x
        self.y = y
        self.n = len(x)
        self.p = 5 + degree # 4 for sinusoid model, deg+1 for baseline
        # self.p = 1 + degree # 4 for sinusoid model, deg+1 for baseline
        self.nb = degree

    
    def create_parameters(self, master):
        """ Set up the parameter class with estimates from the master
        class """

        self.params = params = Parameters()

        # Add parameters for sinusoidal model
        params.add('amplitude', value=master.p0['amplitude'], min=0)
        params.add('period', value=master.p0['period'], min=0)
        params.add('phase', value=master.p0['phase']%(2*np.pi), min=0,
                   max=2*np.pi)
        params.add('decay', value=master.p0['decay'])

        # self.weights = master.p0['weights']
        # # Normalize weights to go from 0.25 -> 1
        # min_weight = 0.25
        # self.weights *= (1 - min_weight)/self.weights.max()
        # self.weights += min_weight

        # assert len(self.weights) == len(self.x), "Weights incorrect"

        # Add parameters for baseline model (polynomial deg=nb)
        b_estimate = np.polyfit(self.x, master.bio.yvals['mean'],
                                self.nb)[::-1]
        for i, par in enumerate(b_estimate):
            params.add('bl'+str(i), value=par)

    def fit(self):
        """ Fit the function to the data """

        self.result = minimize(minimize_function, self.params,
                               args=(self.x, self.y))

        # Add some error checking here?

        # Fitted values
        self.sinusoid = sinusoid_component(self.params, self.x)
        self.baseline = baseline_component(self.params, self.x)
        self.yhat = self.y + self.result.residual

    # def check_outliers(self, result):
    #     """ check the residuals for outliers, if they exist, pop them
    #     and re-run the optimization """

    #     alpha = 0.001
    #     method = 'fdr_bh'
    #     
    #     resid = result.residual
    #     n_resid = len(resid)
    #     df = n_resid - self.p
    #     stats.t.sf(np.abs(resid), df) * 2
    #     unadj_p = stats.t.sf(np.abs(resid), df) * 2
    #     reject, adj_p, acs, acb = multipletests(unadj_p, alpha=alpha, method=method)

    def _ln_l(self):
        """ Get the log-likelyhood of the fitted function """
        return -0.5*self.n*(np.log(2*np.pi) + 1 - np.log(self.n) +
                            np.log((self.result.residual**2).sum()))

    def _aic(self):
        """ Akaike Information Criterion """
        return 2*self.p - 2*self._ln_l()

    def _aic_c(self):
        """ Bias-corrected AIC """
        return self._aic() + 2*self.p*(self.p + 1)/(self.n - self.p - 1)

    def _bic(self):
        """ Bayesian Information Criterion """
        return self.p*np.log(self.n) - 2*self._ln_l()

    def _calc_r2(self):
        SSres = (self.result.residual**2).sum()
        SStot = ((self.y - self.y.mean())**2).sum()
        return 1 - SSres/SStot


def sinusoid_component(params, x):
    amplitude = params['amplitude'].value
    period = params['period'].value
    phase = params['phase'].value
    decay = params['decay'].value

    # Allow for decays to be in both units of 1/hrs or 1/rad
    if sinusoid_component.decay_units == '1/rad':
        decay *= 2*np.pi/period

    return (amplitude * np.cos((2*np.pi/period)*x + phase) *
            np.exp(-decay*x))

# Default decay units (same as Bioluminescence package)
sinusoid_component.decay_units = '1/hrs'


def baseline_component(params, x):
    bl_pars = [params[key].value for key in params.keys() if
               key.startswith('bl')]
    tsum = np.zeros(x.shape)
    for i, par in enumerate(bl_pars): tsum += par*x**i
    return tsum

def minimize_function(params, x, y):
    sinusoid = sinusoid_component(params, x)
    baseline = baseline_component(params, x)
    resid = sinusoid + baseline - y
    return resid



class DecayingSinusoid(object):
    
    def __init__(self, x, y, max_degree=6, outlier_sigma=4, ic='bic',
                 decay_units='1/hrs', specific_degree=False):
        """ Calculate the lowest AICc model for the given x,y data.
        max_degree specifies the maximum degree of the baseline function

        specific_degree=True specifies that only the one model
        corresponding to max_degree should be calculated.
        """
        
        # Pop outlier data points
        x, y = _pop_nans(x, y)
        valid = reject_outliers(y, outlier_sigma)
        self.x = x[valid]
        self.y = y[valid]

        self.max_degree = max_degree

        self.opt = {
            'bio_period_guess'   : 24.,
            'bio_detrend_period' : 24.,
            'selection'          : ic,
            'decay_units'        : decay_units,
            'specific_degree'    : specific_degree,
        }

        # Change default in sinusoid component function
        sinusoid_component.decay_units = decay_units


    def run(self):
        self._estimate_parameters()
        self._fit_models()
        self._calculate_averaged_parameters()
        return self

    def _estimate_parameters(self):
        self.bio = Bioluminescence(self.x, self.y,
                      period_guess=self.opt['bio_period_guess'])
        self.bio.filter()
        self.bio.detrend()
        self.p0 = self.bio.estimate_sinusoid_pars()

        # Bioluminescence returns decay in units of 1/rad, change here
        # to 1/hrs
        if self.opt['decay_units'] != '1/rad':
            self.p0['decay'] *= 2*np.pi/self.p0['period']
        

    def _fit_models(self):
        self.models = []
        start = self.max_degree if self.opt['specific_degree'] else 0
        for i in xrange(start, self.max_degree+1):
            self.models += [SingleModel(self.x, self.y, i)]
            self.models[-1].create_parameters(self)
            self.models[-1].fit()

    def _calculate_model_weights(self):
        if self.opt['selection'].lower() == 'aic':
            ics = np.array([model._aic_c() for model in self.models])
        elif self.opt['selection'].lower() == 'bic':
            ics = np.array([model._bic() for model in self.models])

        del_ics = ics - ics.min()
        return np.exp(-0.5*del_ics)/(np.exp(-0.5*del_ics).sum())

    def _calculate_averaged_parameters(self):
        self.model_weights = model_weights = \
                self._calculate_model_weights()

        param_keys = [model.params.keys() for model in self.models]

        self.averaged_params = {}
        for param in param_keys[-1]:
            self.averaged_params[param] = \
                    ModelAveragedParameter(param, self.models,
                                           model_weights)

        # Shortcut method for easier access
        self.best_model = self.models[self.model_weights.argmax()]

    def _best_model_degree(self):
        return self.models[self.model_weights.argmax()].nb

    def _best_model_r2(self):
        return self.models[self.model_weights.argmax()]._calc_r2()

    def _hilbert_fit(self):
        """ Estimate the decay and amplitude parameters using the
        Bioluminescence module for a second opinion. (Depricated) """
        return (self.p0['amplitude'], self.p0['decay'])

    def report(self):
        print "Fit ({0})".format(self.opt['selection'])
        print "---"
        print "Best interpolation degree: {0}".format(
                self._best_model_degree())
        print "Best R2: {0}".format(self._best_model_r2())
        print ""
        print "Parameters"
        print "----------"
        for key in ['amplitude', 'period', 'phase', 'decay']:
            print "{0:>9}: {1:7.3f} +/- {2:6.3f}".format(key,
                self.averaged_params[key].value,
                1.96*self.averaged_params[key].stderr)




class ModelAveragedParameter(object):
    def __init__(self, key, models, weights):
        """ Calculates the model-averaged value for the parameter
        specified by 'key' eighted by the akaike weights in 'weights'. 
        
        Follows method outlined in:
        Symonds, M. R. E., and Moussalli, A. (2010). A brief guide to
        model selection, multimodel inference and model averaging in
        behavioural ecology using Akaike's information criterion.
        Behavioral Ecology and Sociobiology, 65(1), 13-21.
        doi:10.1007/s00265-010-1037-6
        """

        param_keys = [model.params.keys() for model in models]
        in_models = np.array([key in keys for keys in param_keys])

        self.key = key
        self.models = np.array(models)[in_models].tolist()
        self.weights = weights[in_models]

        means = np.array([model.params[key].value for model in
                          self.models])
        variances = np.array([model.params[key].stderr**2 for model in
                              self.models])

        self.total_weight = self.weights.sum()
        self.value = (self.weights*means).sum()/self.total_weight
        
        tvar = 0
        for weight, var, mean in zip(self.weights, variances, means):
            tvar += weight * np.sqrt(var + (mean - self.value)**2)

        self.stderr = tvar
        self.lb = self.value - 1.96*self.stderr
        self.ub = self.value + 1.96*self.stderr


def reject_outliers(data, m=4):
    return abs(data - np.mean(data)) < m * np.std(data)

def _pop_nans(x, y):
    """ Remove nans from incoming dataset """
    xnan = np.isnan(x)
    ynan = np.isnan(y)
    return x[~xnan & ~ynan], y[~xnan & ~ynan]


class StochasticModelEstimator(object):

    def __init__(self, x, ys, base, **kwargs):
        """ Convenience class to estimate the relevant oscillatory
        parameters from a stochastic-simulated model. Fits a decaying
        sinusoid to each state variable, (ys.shape == (len(x), NEQ)),
        Additional kwargs are passed to the DecayingSinusoid instances

        Takes the expected amplitude for each state variable from the
        cosine components (assuming the stochastic simulation has t=0
        corresponding to the synchronized state). 
        """ 

        assert len(x) == ys.shape[0], "Incorrect Dimensions, x"
        assert base.NEQ == ys.shape[1], "Incorrect Dimensions, y"

        self.x = x
        self.ys = ys
        self._kwargs = kwargs
        
        amp, phase, baseline = base._cos_components()
        self._cos_dict = {
            'amp'      : amp,
            'phase'    : phase,
            'baseline' : baseline,
        }

        self.masters = [self._run_single_state(i) for i in
                        xrange(base.NEQ)]

        param_keys = ['decay', 'period']

        self.params = {}
        for param in param_keys:
            vals = np.array([master.averaged_params[param].value for
                             master in self.masters])
            self.params[param] = np.average(vals)

    def _run_single_state(self, i):

        imaster = DecayingSinusoid(self.x, self.ys[:,i], max_degree=0,
                                   **self._kwargs)
        imaster._estimate_parameters()
        imaster.models = [SingleModel(imaster.x, imaster.y, 1)]
        imodel = imaster.models[0]
        imodel.create_parameters(imaster)
        imodel.params['amplitude'].value = self._cos_dict['amp'][i]
        imodel.params['amplitude'].vary = False
        imodel.fit()
        imaster._fit_models()
        imaster._calculate_averaged_parameters()
        return imaster


if __name__ == "__main__":
    import sys

    x = np.array([
          5.52408,    6.07656,    6.62928,    7.18176,    7.73472,
          8.28816,    8.84088,    9.3924 ,    9.94488,   10.49784,
         11.05008,   11.60184,   12.15456,   12.70704,   13.25928,
         13.812  ,   14.36472,   14.9172 ,   15.46992,   16.02384,
         16.57656,   17.12832,   17.68104,   18.23328,   18.78528,
         19.33656,   19.88856,   20.44056,   20.99088,   21.54336,
         22.09464,   22.64592,   23.19672,   23.74776,   24.29952,
         24.85032,   25.40232,   25.95336,   26.50536,   27.05688,
         27.60768,   28.15968,   28.71144,   29.26224,   29.814  ,
         30.36504,   30.91608,   31.46832,   32.01936,   32.57232,
         33.12288,   33.67392,   34.22496,   34.77576,   35.32608,
         35.87712,   36.4284 ,   36.9792 ,   37.53024,   38.08224,
         38.63424,   39.18576,   39.73704,   40.28952,   40.842  ,
         41.394  ,   41.94552,   42.49776,   43.04952,   43.6008 ,
         44.15112,   44.70168,   45.25248,   45.80472,   46.356  ,
         46.90752,   47.4588 ,   48.01104,   48.564  ,   49.11576,
         49.66704,   50.21856,   50.76912,   51.32064,   51.8712 ,
         52.42248,   52.9728 ,   53.5236 ,   54.07488,   54.6264 ,
         55.17768,   55.72872,   56.28048,   56.83296,   57.38496,
         57.93672,   58.488  ,   59.03856,   59.58864,   60.1392 ,
         60.69048,   61.24104,   61.79232,   62.34312,   62.89416,
         63.44544,   63.9972 ,   64.54896,   65.09976,   65.65032,
         66.20232,   66.75312,   67.30344,   67.85352,   68.40384,
         68.95536,   69.50592,   70.05672,   70.60776,   71.15856,
         71.70936,   72.26016,   72.81168,   73.36248,   73.91304,
         74.4648 ,   75.01584,   75.5676 ,   76.11816,   76.66872,
         77.2188 ,   77.76912,   78.32016,   78.87072,   79.42176,
         79.97352,   80.52576,   81.07728,   81.62832,   82.17984,
         82.73064,   83.28144,   83.83224,   84.38304,   84.93456,
         85.48464,   86.03544,   86.5872 ,   87.138  ,   87.68952,
         88.24176,   88.79328,   89.34408,   89.89464,   90.44592,
         90.9972 ,   91.54752,   92.09808,   92.64936,   93.19992,
         93.75168,   94.30368,   94.85472,   95.40576,   95.95728,
         96.50928,   97.06224,   97.61472,   98.166  ,   98.71704,
         99.26808,   99.82008,  100.3716 ,  100.92336,  101.4744 ,
        102.02664,  102.57768,  103.1292 ,  103.68   ,  104.23104,
        104.7828 ,  105.33504,  105.8856 ,  106.43712,  106.9884 ,
        107.53968,  108.09024,  108.642  ,  109.1928 ,  109.7436 ,
        110.2956 ,  110.84688,  111.39864,  111.95064,  112.50168,
        113.05464,  113.60592,  114.15672,  114.70824,  115.25976,
        115.81032,  116.36136,  116.91216,  117.4644 ,  118.01592,
        118.5672 ,  119.11872,  119.66976,  120.2208 ,  120.7728 ,
        121.3248 ,  121.87608,  122.42784,  122.97912,  123.53016,
        124.08072,  124.63176,  125.1828 ,  125.7336 ,  126.2844 ,
        126.83616,  127.3872 ,  127.938  ,  128.48952,  129.04248,
        129.5952 ,  130.14648,  130.69824,  131.24976,  131.80032,
        132.35136,  132.90312,  133.45464,  134.00592,  134.55816,
        135.11064,  135.66168,  136.21392,  136.76496,  137.3172 ,
        137.86824,  138.42048,  138.97224,  139.52352,  140.07408,
        140.62536,  141.17544,  141.72696,  142.278  ,  142.82976,
        143.3808 ,  143.93232,  144.4836 ,  145.03536,  145.58712,
        146.13816,  146.6892 ,  147.24072,  147.79224,  148.34352,
        148.8948 ,  149.44608,  149.99808,  150.54936,  151.10088,
        151.65192,  152.20344,  152.75496,  153.30744,  153.85872,
        154.40976,  154.96128,  155.5128 ,  156.06408,  156.6156 ,
        157.16664,  157.71768,  158.26896,  158.82   ,  159.37128,
        159.92256,  160.4748 ,  161.02656,  161.5788 ,  162.13008,
        162.68256,  163.23384,  163.78536,  164.33616,  164.88744,
        165.43896,  165.99024])

    y = np.array([
        13321.,  12971.,  13046.,  12725.,  13046.,  13098.,  13783.,
        14342.,  14990.,  15698.,  16786.,  17762.,  18164.,  18768.,
        19334.,  20414.,  20846.,  20727.,  21629.,  21733.,  21934.,
        21994.,  21584.,  21569.,  21673.,  21062.,  19975.,  20302.,
        19207.,  18924.,  18335.,  17158.,  16369.,  15266.,  14901.,
        13724.,  13128.,  11734.,  11645.,  10952.,  10393.,  10207.,
         9730.,   9343.,   9082.,   8948.,   8881.,   8851.,   9112.,
         9447.,   9715.,   9879.,  10691.,  10855.,  11429.,  12010.,
        12770.,  13537.,  14074.,  14931.,  15199.,  16205.,  16361.,
        16793.,  17359.,  17739.,  18209.,  18544.,  18283.,  18298.,
        18037.,  18008.,  17620.,  17374.,  17441.,  16965.,  16272.,
        15951.,  15415.,  15214.,  15013.,  14580.,  14305.,  13932.,
        13619.,  13090.,  13046.,  12867.,  12614.,  12919.,  12755.,
        12658.,  12569.,  12755.,  13031.,  13083.,  13590.,  13858.,
        14290.,  14253.,  14700.,  14908.,  15340.,  15758.,  16071.,
        16570.,  17158.,  17561.,  17806.,  18380.,  18306.,  18872.,
        18686.,  19162.,  19006.,  19289.,  19274.,  19677.,  19356.,
        19215.,  19662.,  19624.,  19259.,  19140.,  18917.,  18835.,
        18559.,  18470.,  18775.,  18552.,  18164.,  18365.,  18283.,
        18373.,  18306.,  18544.,  18738.,  18917.,  18522.,  18492.,
        19066.,  19379.,  19468.,  20168.,  20533.,  19736.,  21055.,
        21211.,  21457.,  21532.,  21837.,  22441.,  22940.,  23245.,
        23215.,  23156.,  23506.,  24184.,  23849.,  24363.,  24720.,
        24400.,  24661.,  24758.,  24512.,  24333.,  24884.,  25458.,
        25108.,  24832.,  25711.,  24862.,  24951.,  24974.,  24951.,
        25764.,  25510.,  25935.,  25361.,  25354.,  26151.,  25913.,
        26017.,  26732.,  26941.,  27201.,  27060.,  27187.,  26285.,
        28319.,  28550.,  29057.,  29012.,  29422.,  29481.,  29817.,
        30636.,  30472.,  31448.,  31731.,  32074.,  32350.,  32789.,
        32223.,  32618.,  32543.,  33333.,  33862.,  33430.,  33646.,
        33341.,  34235.,  33728.,  34302.,  34682.,  34458.,  33467.,
        34220.,  34287.,  34838.,  34637.,  34734.,  35352.,  35583.,
        35345.,  36120.,  35315.,  35956.,  35322.,  36500.,  36790.,
        36872.,  36947.,  36790.,  36425.,  37103.,  37066.,  37602.,
        38407.,  38474.,  38846.,  38139.,  38243.,  38645.,  38936.,
        39919.,  39711.,  40210.,  40061.,  39897.,  40836.,  39986.,
        40709.,  41275.,  40642.,  41171.,  41335.,  41283.,  41923.,
        41231.,  40590.,  41395.,  41231.,  40985.,  40366.,  41089.,
        41581.,  41752.,  40754.,  42073.,  41156.,  40754.,  40985.,
        41223.,  41558.,  40731.,  40195.,  41007.,  41737.,  40791.,
        40858.,  41640.,  40590.,  40687.,  41730.,  41208.,  39733.,
        41447.,  40530.,  41357.,  41000.,  39115.])


    # x = np.arange(0, 74, 2)
    # y = np.array([ -93.376, -128.174, -115.173,  -46.591,   35.161,
    #               92.173,  133.255,  141.447,  133.079,   68.621,
    #               11.983,  -32.145,  -57.393,  -68.721,  -60.759,
    #               -44.467,   -9.305,   21.757,   49.879,   64.551,
    #               52.803,   33.005,    2.747,  -16.561,  -37.12 ,
    #               -48.248,  -48.866,  -44.504,  -28.652,  -15.72 ,
    #               -2.928,   11.604,   16.506,   16.048,   10.81 ,
    #               3.272, np.nan])

    # import pandas as pd
    # data = pd.read_pickle('../decay/Hogenesch_data/genome_scale.p')
    # ts = np.arange(0, 74, 2)
    # # x = np.array(ts, dtype=float)
    # trange = [str(t) for t in ts] 

    # try: index = int(sys.argv[1])
    # except (IndexError, ValueError): index = 0

    # row = data.iloc[index]

    # y = np.array(row[trange].values, dtype=float)

    master = DecayingSinusoid(x[3:], y[3:], max_degree=4,
                              decay_units='1/hrs',
                              specific_degree=True).run()
    master.report()
    print "Expected Phase Decay: {0:0.3f}".format(
        master.averaged_params['period'].value *
        master.averaged_params['decay'].value / (2*np.pi))

    master = DecayingSinusoid(x, y, max_degree=4).run()
    master.report()
    master.opt['selection'] = 'aic'
    master._calculate_averaged_parameters()
    print ''
    master.report()


    if 'plot' in sys.argv:
        from CommonFiles.PlotOptions import PlotOptions, layout_pad
        PlotOptions(uselatex=True)
        import matplotlib.pyplot as plt
        master.opt['selection'] = 'bic'
        master._calculate_averaged_parameters()
        i = master._best_model_degree()
        model = master.models[i]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(master.x, master.y, '.', zorder=2)
        ax.plot(model.x, model.yhat, '-.')
        ax.plot(model.x, model.baseline, '--')
        fig.tight_layout(**layout_pad)
        plt.show()



    # sub = SingleModel(master.x, master.y, 5)
    # sub.create_parameters(master)
    # sub.fit()

    # from CommonFiles.PlotOptions import color_range, blue, red, PlotOptions, layout_pad
    # # import matplotlib
    # PlotOptions(uselatex=True)
    # import matplotlib.pyplot as plt

    # fig, axmatrix = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    # for i, ax in enumerate(axmatrix.flatten()):
    #     sub = SingleModel(master.x, master.y, i)
    #     sub.create_parameters(master)
    #     sub.fit()
    #     ax.plot(x, y, '.', zorder=2)
    #     ax.plot(sub.x, sub.sinusoid, '--')
    #     ax.plot(sub.x, sub.yhat, '-.')
    #     ax.plot(sub.x, sub.baseline, '-')
    #     ax.text(0.9, 0.9, r'$p = ' + str(i) + r'$',
    #             horizontalalignment='right', verticalalignment='top',
    #             transform=ax.transAxes)

    # ax.set_xlim([0, sub.x.max()])
    # ax.set_ylim([-150, 150])
    # axmatrix[1, 1].set_xlabel('Time (hr)')
    # fig.tight_layout(**layout_pad)


    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # # colors = list(color_range(len(master.models) + 2, cm=matplotlib.cm.BuPu))[2:]
    # 
    # # for model, color in zip(master.models, colors):
    # #     ax.plot(model.x, model.yhat, color=color)
    # # ax.plot(x, y, '.')
    # # ax.plot(sub.x, sub.baseline, '-')
    # # ax.plot(sub.x, sub.sinusoid, '--')
    # # ax.plot(sub.x, sub.yhat, ':')
    # plt.show()

