# Model of a hysteresis-driven negative-feedback oscillator, taken from
# B. Novak and J. J. Tyson, "Design principles of biochemical
# oscillators," Nat. Rev. Mol. Cell Biol., vol. 9, no. 12, pp. 981-91,
# Dec. 2008.
# Figure 3, equation 8.


# This model has a very short period (~3), so routines should be run
# with a lower transient time

import numpy as np
import casadi as cs

# Model Constants
paramset = [2., 20., 1., 0.005, 0.05, 0.1]

y0in = [ 0.63590189,  0.7566833 ,  3.51572439] # [X[0], Y[0], Period]

NEQ = 2
NP = 6

def model():
    # Variable Assignments
    X = cs.ssym("X")
    Y = cs.ssym("Y")

    y = cs.vertcat([X,Y]) # vector version of y
    
    # Parameter Assignments
    P  = cs.ssym("P")
    kt = cs.ssym("kt")
    kd = cs.ssym("kd")
    a0 = cs.ssym("a0")
    a1 = cs.ssym("a1")
    a2 = cs.ssym("a2")
        
    symparamset = cs.vertcat([P, kt, kd, a0, a1, a2])
    
    # Time Variable (typically unused for autonomous odes)
    t = cs.ssym("t")
    
    
    ode = [[]]*NEQ
    ode[0] = 1 / (1 + Y**P) - X
    ode[1] = kt*X - kd*Y - Y/(a0 + a1*Y + a2*Y**2)
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Simple Hysteresis model")
    
    return fn

def create_class():
    from CommonFiles.pBase import pBase
    return pBase(model(), paramset, y0in)


# John's files for stochasic evaluation
import CommonFiles.stochkit_resources as stk
import CommonFiles.modelbuilder as mb

def SSAModel(base, vol, nrep=1, y0_pop=None):
    """ nrep repeats the model (uncoupled system) to allow for the
    passing of a matrix of initial values. y0_pop should be a matrix of
    size (NREP x NEQ) """

    if y0_pop is None:
        y0_pop = np.tile((vol*base.y0[:-1]), (nrep, 1)).astype(int)
    else: y0_pop = vol*y0_pop

    stoch_fn = stk.StochKitModel(
        name=base.model.getOption('name') + '_stoch_' + str(nrep))

    bd = mb.SSA_builder(
        mb.SSA_builder.expand_labelarray(base.ylabels, nrep),
        mb.SSA_builder.expand_labelarray(base.plabels, nrep),
        y0_pop.flatten(order='C'),
        np.tile(base.paramset, (nrep, 1)).flatten(order='C'),
        stoch_fn, vol)

    for index in xrange(nrep):
        si = '_' + str(index)

        bd.SSA_tyson_x('x prod nonlinear term'+si, 'X'+si, 'Y'+si,
                       'P'+si)
        bd.SSA_MA_deg('x degradation'+si, 'X'+si, '1')
        bd.SSA_MA_tln('y creation'+si, 'Y'+si, 'kt'+si, 'X'+si)
        bd.SSA_MA_deg('y degradation linear'+si, 'Y'+si, 'kd'+si)
        bd.SSA_tyson_y('y nonlinear term'+si, 'Y'+si, 'a0'+si, 'a1'+si,
                       'a2'+si)

    return stoch_fn


def simulate_stoch(base, vol, t=None, ntraj=1, increment=0.01,
                   job_id='', y0_pop=None, nrep=1):

    stoch_fn = SSAModel(base, vol, nrep=nrep, y0_pop=y0_pop)
    trajectories = stk.stochkit(stoch_fn, job_id='tysonmodel' +
                                str(job_id), t=t,
                                number_of_trajectories=ntraj,
                                increment=increment)

    ts = trajectories[0][:,0]
    traj = np.array(trajectories)[:,:,1:]

    # Restructure trajectory array to recreate (replicate, time, state)
    # indexing scheme
    traj = traj.reshape((ntraj, len(ts), nrep, base.NEQ)) # Expand
    traj = traj.swapaxes(2,1) # Bring reps and traj to front
    # Collapse to single axis
    traj = traj.reshape((nrep*ntraj,len(ts), base.NEQ)) 
    traj = traj/vol # return concentrations

    return ts, traj


if __name__ == "__main__":

    base_control = create_class()
    base_control.limitCycle()

    # Calculate stochastic trajectories
    vol = 200
    ts, traj = simulate_stoch(base_control, vol,
                              t=5*base_control.y0[-1], nrep=100,
                              increment=base_control.y0[-1]/100)

    # Plot trajectories
    import matplotlib.pyplot as plt
    from CommonFiles.PlotOptions import PlotOptions, layout_pad
    PlotOptions(uselatex=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts, traj[:,:,0].T, 'k', alpha=0.01, zorder=1)
    ax.plot(ts, traj.mean(0)[:,0], zorder=3)
    ax.plot(ts, base_control.lc(ts)[:,0], '--', zorder=2)
    ax.set_rasterization_zorder(1)
    fig.tight_layout(**layout_pad)

    plt.show()
