import numpy as np
import scipy as sc
import casadi as cs
import scipy.optimize as opt

from CommonFiles.pBase import pBase

class AmplitudeResponse(pBase):

    def _create_arc_integrator(self, trans_duration=3):
        """ Create integrator and simulator objects for later use """

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
        self.arc_traj_ts = np.linspace(0, tf, num=int(100*trans_duration),
                                  endpoint=True) 
        self.arcsim = cs.Simulator(self.arcint, self.arc_traj_ts/tf)
        self.arcsim.init()


    def _sARC_single_time(self, state, tstart, amount, trans_duration):
        """ Compares a single perturbation of state to a reference
        trajectory """

        # Find y0 at start of pulse
        y0 = self.lc(tstart)
        y0[state] += amount
        return self._simulate_pulse_and_ref(y0, tstart, trans_duration)



    def _simulate_pulse_and_ref(self, y0, tstart, trans_duration):
        """ Simulate the pulse and match with the limit cycle to find
        the amplitude and phase response """
        self.arcsim.setInput(y0, cs.INTEGRATOR_X0)
        self.arcsim.setInput(list(self.paramset) +
                             [self.y0[-1]*trans_duration],
                             cs.INTEGRATOR_P)
        self.arcsim.evaluate()
        trajectory = self.arcsim.output().toArray()

        yend = trajectory[-1]
        tend = self.arc_traj_ts[-1]%self.y0[-1] + tstart

        def resy(t):
            return np.linalg.norm(yend - self.lc(t%self.y0[-1]))

        # Minimize resy(t)
        tvals = np.linspace(0, self.y0[-1], num=25)
        tguess = tvals[np.array([resy(t) for t in tvals]).argmin()]
        tmin = opt.fmin(resy, tguess, disp=0)[0]%self.y0[-1]
        assert resy(tmin)/self.NEQ < 1E-3, "transient not converged"

        if tmin > self.y0[-1]/2: tmin += -self.y0[-1]

        tdiff = tmin-tend

        # rescale tdiff from -T/2 to T/2
        tdiff = tdiff%self.y0[-1]
        if tdiff > self.y0[-1]/2: tdiff += -self.y0[-1]

        reference = self.lc((self.arc_traj_ts + tend + tdiff)%self.y0[-1])

        return trajectory, reference, tdiff


    def calc_sARC_finite_difference(self, state, amount,
                                    trans_duration=3, res=100):
        """ Find ARC using finite difference methods """

        from scipy.integrate import cumtrapz
        self._create_arc_integrator(trans_duration)
        if not hasattr(self, 'avg') : self.average()
        if not hasattr(self, 'lc')  : self.limitCycle()

        times = np.linspace(0, self.y0[-1], num=res)
        trajectories = []
        references = []
        arc = []
        prc = []

        for time in times:
            out = self._sARC_single_time(state, time, amount,
                                         trans_duration)
            traj, ref, phase = out
            trajectories += [traj]
            references += [ref]
            prc += [phase]
            amp = cumtrapz(((traj - self.avg)**2
                            - (ref - self.avg)**2).T, self.arc_traj_ts).T
            arc += [amp[-1]]

        trajectories = np.array(trajectories) 
        references = np.array(references)
        prc = np.array(prc) # Phase response curve
        arc = np.array(arc) # Amplitude response curve

        return times, arc/amount, prc/amount
 


    def calc_pARC_finite_difference(self, param, amount, pulse_duration,
                                    trans_duration=3, res=100, rel=False):
        """ Find ARC using finite difference methods """

        from scipy.integrate import cumtrapz
        self._create_arc_integrator(trans_duration)
        if not hasattr(self, 'avg') : self.average()
        if not hasattr(self, 'lc')  : self.limitCycle()

        try: par_ind = self.pdict[param]
        except KeyError: par_ind = int(param)

        times = np.linspace(0, self.y0[-1], num=res)
        trajectories = []
        references = []
        arc = []
        prc = []

        for time in times:
            out = self._pARC_single_time(par_ind, time, amount,
                                         pulse_duration, trans_duration)
            traj, ref, phase = out
            trajectories += [traj]
            references += [ref]
            prc += [phase]
            amp = cumtrapz(((traj - self.avg)**2
                            - (ref - self.avg)**2).T, self.arc_traj_ts).T
            arc += [amp[-1]]

        trajectories = np.array(trajectories) 
        references = np.array(references)
        prc = np.array(prc)/(amount*pulse_duration) # Phase response curve
        arc = np.array(arc)/(amount*pulse_duration) # Amplitude

        if rel: # Relative curves?
            prc *= self.paramset[par_ind]
            arc *= self.paramset[par_ind]/self.avg

        return (times, arc, prc)


        
    def _pARC_single_time(self, par_ind, tstart, amount, pulse_duration,
                          trans_duration):

        # Find conditions at start of pulse
        y0 = self.lc(tstart)
        param_init = np.array(self.paramset)
        param_init[par_ind] += amount

        # Integrate trajectory through pulse
        # Find parameter set for pulse
        self.arcint.setInput(y0, cs.INTEGRATOR_X0)
        self.arcint.setInput(param_init.tolist() + [pulse_duration],
                             cs.INTEGRATOR_P)
        self.arcint.evaluate()
        yf = np.array(self.arcint.output())
        
        return self._simulate_pulse_and_ref(yf, tstart+pulse_duration,
                                            trans_duration)


    
    


if __name__ == "__main__":

    from CommonFiles.Models.tyson2statemodel import model, paramset
    import matplotlib.pylab as plt

    test = AmplitudeResponse(model(), paramset)
    test.findPRC(200)
    # ts, s_arc_diff = test.findSARC(0)
    # ts, s_arc, s_prc = test.calc_sARC_finite_difference(0, 0.01)

    # plt.figure()
    # plt.plot(ts, s_arc[:,0], '.:')
    # plt.plot(ts, s_arc_diff[:,0], 'k-')
    # plt.title('s_ARC')
    # plt.figure()
    # plt.plot(ts, s_prc, '.:')
    # plt.plot(ts, test.sIPRC_interp(ts)[:,0], 'k-')
    # plt.title('s_PRC')
    
    arc_fig = plt.figure()
    prc_fig = plt.figure()

    arc_ax = arc_fig.add_subplot(111)
    prc_ax = prc_fig.add_subplot(111)


    arc_ax.set_title('p_ARC')
    prc_ax.set_title('p_PRC')

    arc_ax.set_xlim([0, test.y0[-1]])
    prc_ax.set_xlim([0, test.y0[-1]])

    sARC = np.array([test.findSARC(state)[1] for state in xrange(test.NEQ)])


    
    duration = 0.5
    amount = 0.5
    par_ind = 2
    ts, p_arc, p_prc = test.calc_pARC_finite_difference(par_ind, amount,
                                                        duration)

    ts_diff, p_arc_diff = test.findPARC(par_ind)
    arc_ax.plot(ts, p_arc[:,1], '.:')
    arc_ax.plot(ts_diff, p_arc_diff[:,1], 'k-')
    prc_ax.plot(ts, p_prc, '.:')
    prc_ax.plot(test.prc_ts, test.pPRC[:,par_ind], 'k-')

    lega = arc_ax.legend(['FD', 'Sensitivity'])
    lega.draw_frame(False)
    legp = prc_ax.legend(['FD', 'Sensitivity'])
    legp.draw_frame(False)

    arc_ax.axhline(0, ls='--', color='grey', zorder=0)
    prc_ax.axhline(0, ls='--', color='grey', zorder=0)


    plt.show()



