import numpy as np

from CommonFiles.Bioluminescence import Bioluminescence

from lmfit import minimize, Parameters

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

        # Add parameters for baseline model (polynomial deg=nb)
        # b_estimate = np.polyfit(self.x, self.y, self.nb)[::-1]
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

    return (amplitude * np.cos((2*np.pi/period)*x + phase) *
            np.exp(-2*np.pi*decay*x/period))

def baseline_component(params, x):
    bl_pars = [params[key].value for key in params.keys() if
               key.startswith('bl')]
    tsum = np.zeros(x.shape)
    for i, par in enumerate(bl_pars): tsum += par*x**i
    return tsum

def minimize_function(params, x, y):
    sinusoid = sinusoid_component(params, x)
    baseline = baseline_component(params, x)
    return sinusoid + baseline - y



class DecayingSinusoid(object):
    
    def __init__(self, x, y, max_degree=6, outlier_sigma=4, ic='bic'):
        """ Calculate the lowest AICc model for the given x,y data.
        max_degree specifies the maximum degree of the baseline function
        """
        
        # Pop outlier data points
        x, y = _pop_nans(x, y)
        valid = reject_outliers(y, outlier_sigma)
        self.x = x[valid]
        self.y = y[valid]

        self.max_degree = max_degree

        self.opt = {
            'bio_period_guess' : 24.,
            'bio_detrend_period' : 24.,
            'selection' : ic,
        }

    def run(self):
        self._estimate_parameters()
        self._fit_models()
        self._calculate_averaged_parameters()
        return self

    def _estimate_parameters(self):
        self.bio = Bioluminescence(self.x, self.y,
                      period_guess=self.opt['bio_period_guess'])
        self.bio.detrend(detrend_period=self.opt['bio_detrend_period'])
        self.p0 = self.bio.estimate_sinusoid_pars()

    def _fit_models(self):
        self.models = []
        for i in xrange(self.max_degree+1):
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



if __name__ == "__main__":
    import sys

    # x = np.arange(0, 74, 2)
    # y = np.array([ -93.376, -128.174, -115.173,  -46.591,   35.161,
    #               92.173,  133.255,  141.447,  133.079,   68.621,
    #               11.983,  -32.145,  -57.393,  -68.721,  -60.759,
    #               -44.467,   -9.305,   21.757,   49.879,   64.551,
    #               52.803,   33.005,    2.747,  -16.561,  -37.12 ,
    #               -48.248,  -48.866,  -44.504,  -28.652,  -15.72 ,
    #               -2.928,   11.604,   16.506,   16.048,   10.81 ,
    #               3.272, np.nan])

    import pandas as pd
    data = pd.read_pickle('../decay/Hogenesch_data/genome_scale.p')
    ts = np.arange(0, 74, 2)
    x = np.array(ts, dtype=float)
    trange = [str(t) for t in ts] 

    try: index = int(sys.argv[1])
    except (IndexError, ValueError): index = 0

    row = data.iloc[index]

    y = np.array(row[trange].values, dtype=float)

    master = DecayingSinusoid(x[3:], y[3:], max_degree=4).run()
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

