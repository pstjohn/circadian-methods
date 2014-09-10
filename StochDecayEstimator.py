import numpy as np

from lmfit import minimize, Parameters

from CommonFiles.Amplitude import Amplitude, gaussian_phase_distribution
from CommonFiles.DecayingSinusoid import DecayingSinusoid, SingleModel




class StochDecayEstimator(object):

    def __init__(self, x, ys, base, vary_amp=False, **kwargs):
        """ Class to estimate the decay rate (phase diffusivity) for a
        stochastic model simulation. Assumes the population starts
        completely synchronized at y[0] == base.y0[:-1] """

        assert len(x) == ys.shape[0], "Incorrect Dimensions, x"
        assert base.NEQ == ys.shape[1], "Incorrect Dimensions, y"

        self.x = x
        self.ys = ys
        self.base = base
        self._kwargs = kwargs
        self.vary_amp = vary_amp

        self.base.__class__ = Amplitude
        self.base._init_amp_class()

        amp, phase, baseline = base._cos_components()
        self._cos_dict = {
            'amp'      : amp,
            'phase'    : phase,
            'baseline' : baseline,
        }

        self.masters = [self._run_single_state(i) for i in
                        xrange(base.NEQ)]

        sinusoid_param_keys = ['decay', 'period']

        self.sinusoid_params = {}
        for param in sinusoid_param_keys:
            vals = np.array([master.averaged_params[param].value for
                             master in self.masters])
            self.sinusoid_params[param] = np.average(vals)


        xbar_param = Parameters()
        xbar_param.add('decay', value=self.sinusoid_params['decay'],
                       min=0)
        self.result = minimize(self._minimize_function, xbar_param)
        self.decay = xbar_param['decay'].value
        self.x_bar = self._calc_x_bar(xbar_param)
        ss_res = ((self.ys - self.x_bar)**2).sum()
        ss_tot = ((self.ys - self.ys.mean(0))**2).sum()
        self.r2 = 1 - ss_res/ss_tot
        

    def _run_single_state(self, i):

        imaster = DecayingSinusoid(self.x, self.ys[:,i], max_degree=0,
                                   outlier_sigma=10, **self._kwargs)
        imaster._estimate_parameters()
        imaster.models = [SingleModel(imaster.x, imaster.y, 1)]
        imodel = imaster.models[0]
        imodel.create_parameters(imaster)
        imodel.params['amplitude'].value = self._cos_dict['amp'][i]
        imodel.params['amplitude'].vary = self.vary_amp
        imodel.fit()
        imaster._fit_models()
        imaster._calculate_averaged_parameters()
        return imaster

        
    def _calc_x_bar(self, param):
        """ Calculate an estimated x_bar given a guess for the phase
        diffusivity, in units of 1/hrs """

        d_hrs = param['decay'].value
        d_rad = d_hrs * self.base.y0[-1] / (2 * np.pi)

        # Initial population starts with mu, std = 0
        phase_pop = gaussian_phase_distribution(0., 0., d_rad)
        self.base.phase_distribution = phase_pop

        return self.base.x_bar(2 * np.pi * self.x /
                               self.sinusoid_params['period'])


    def _minimize_function(self, param):
        """ Function to minimize via lmfit """
        return (self.ys - self._calc_x_bar(param)).flatten()




if __name__ == "__main__":
    from CommonFiles.Models.Oregonator import create_class, simulate_stoch
    base_control = create_class()

    vol = 0.1
    periods = 7
    ntraj = 100

    ts_c, traj_control = simulate_stoch(base_control, vol,
                                        t=periods*base_control.y0[-1],
                                        ntraj=ntraj,
                                        increment=base_control.y0[-1]/100)

    Estimator = StochDecayEstimator(ts_c, traj_control.mean(0),
                                    base_control)


    import matplotlib.pylab as plt
    plt.plot(ts_c, traj_control.mean(0))
    plt.gca().set_color_cycle(None)
    plt.plot(ts_c, Estimator.x_bar, '--')

    plt.show()

