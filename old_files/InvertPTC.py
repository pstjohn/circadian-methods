import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, fitpack

import matplotlib.pylab as plt

from CommonFiles.Utilities import PeriodicSpline, fnlist, ptc_from_prc


class InvertedPTC(object):
    """ Class to invert a phase transition curve y(x) to return x(y) """

    def __init__(self, phis, prc):

        self.prc_interp = PeriodicSpline(phis, prc, k=4)
        self.ptc_interp = ptc_from_prc(self.prc_interp)

        res = len(phis)

        self.prc_deriv = self.prc_interp.derivative()
        self.div_points = self.prc_deriv.root_offset(-1)

        # Rescale phis and points such that phi=0 is the first
        # division
        offset = self.div_points[0]
        # phi_scaled = (phis - offset)%2*np.pi




        # self.phi_to_theta = lambda phi: (phi - offset)%(2*np.pi)
        # self.theta_to_phi = lambda theta: (theta + offset)%(2*np.pi)
        self.div_scaled = self.div_points - offset

        # Add 2pi to the end of region for final segment
        self.div_scaled = np.hstack([self.div_scaled, 2*np.pi])

        self.fwd_sections = fnlist([])
        self.inv_sections = fnlist([])
        for i, point in enumerate(self.div_scaled[1:]):
            start = self.div_scaled[i]
            stop = point
            section_length = stop - start
            section_res = int(res*section_length/(2*np.pi) + 20)
            x = np.linspace(start, stop, num=section_res,
                            endpoint=True)
            y = x + self.prc_interp(x + offset)
            fwd_spline = BoundedUniveriateSpline(x, y, bbox=[x.min(),
                                                             x.max()])
            inv_spline = BoundedUniveriateSpline(y, x, bbox=[y.min(),
                                                             y.max()])
            

            self.fwd_sections += [fwd_spline]
            self.inv_sections += [inv_spline]


class BoundedUniveriateSpline(InterpolatedUnivariateSpline):

    def __call__(self, x, nu=0):

        xb, xe = self._data[3:5]
        xb += -1E-8
        xe += 1E-8
        x = np.asarray(x)
        y = fitpack.splev(x, self._eval_args, der=nu)

        valid = np.all(np.vstack([x <= xe, x >= xb]), axis=0)
        y[~valid] = np.nan

        return y





if __name__ == "__main__":

    from CommonFiles.Amplitude2 import Amplitude
    from CommonFiles.Models.tyson2statemodel import model, paramset


    amp = Amplitude(model(), paramset)
    state_pulse_creator = amp._s_pulse_creator(1, 0.5)
    amp.calc_pulse_responses(state_pulse_creator)

    res = len(amp.phis)
    phis = amp.phis
    prc = np.hstack([amp.prc_single_cell[:-1][(res/2):],
                     amp.prc_single_cell[:-1][:(res/2)]])

    prc = np.hstack([prc, prc[0]])

    prc_interp = PeriodicSpline(phis, prc)
    ptc = ptc_from_prc(prc_interp)


    # ax.plot(test.ptc_interp(phis), phis, '.')

    test = InvertedPTC(phis, prc)

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(phis, ptc(phis))
    ax.plot(test.div_points, ptc(test.div_points), 'o')
    
    # thetas = test.phi_to_theta(phis)


    # ax.plot(test.div_scaled[:-1], test.ptc_interp(test.div_points), 'o')
    # ax.plot(thetas, test.monotonic_sections(thetas).T)


    plt.show()

    




