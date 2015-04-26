"""
A model of the Goodwin oscillator, using the form specified in B. Ananthasubramaniam, H. Herzel, PLoS One. 9, e104761 (2014). Should show oscillations with Da = Di = Dr, h > 8.
"""



import numpy as np
import casadi as cs


paramset = [10., .1, .1, .1]
y0in = [0.05040051, 0.24366089, 1.69643437, 39.67791427]

NEQ = 3
NP = 4

def model():
    """ Deterministic model """

    # State Variables
    X1 = cs.ssym('X1')
    X2 = cs.ssym('X2')
    X3 = cs.ssym('X3')

    y = cs.vertcat([X1, X2, X3])


    # Parameters
    h = cs.ssym('h')
    a = cs.ssym('a')
    i = cs.ssym('i')
    r = cs.ssym('r')

    symparamset = cs.vertcat([h, a, i, r])


    # Time
    t = cs.ssym('t')


    # ODES
    ode = [[]]*NEQ
    ode[0] = 1./(1. + X3**h) - a*X1
    ode[1] = X1 - i*X2
    ode[2] = X2 - r*X3
    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name", "Goodwin")
    
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
        mb.SSA_builder.expand_labelarray(base.plabels[1:], nrep),
        y0_pop.flatten(order='C'),
        np.tile(base.paramset, (nrep, 1)).flatten(order='C'),
        stoch_fn, vol)

    # ADD REACTIONS
    # =============
    #
    for index in xrange(nrep):
        si = '_' + str(index)

        bd.SSA_MM_P_Goodwin('X1 Production'+si,
                            Prod='X1'+si, Rep='X3'+si,
                            P=int(base.paramset[base.pdict['h']]))

        bd.SSA_MA_tln('X2 Production'+si, 'X2'+si, '1', 'X1'+si)
        bd.SSA_MA_tln('X3 Production'+si, 'X3'+si, '1', 'X2'+si)

        bd.SSA_MA(name="X1 Deg" + si,
                  reactants={bd.species_array[bd.ydict['X1' + si]]:1},
                  rate=bd.param_array[bd.pdict['a' + si]])

        bd.SSA_MA(name="X2 Deg" + si,
                  reactants={bd.species_array[bd.ydict['X2' + si]]:1},
                  rate=bd.param_array[bd.pdict['i' + si]])

        bd.SSA_MA(name="X3 Deg" + si,
                  reactants={bd.species_array[bd.ydict['X3' + si]]:1},
                  rate=bd.param_array[bd.pdict['r' + si]])


    return stoch_fn


def simulate_stoch(base, vol, t=None, ntraj=1, increment=0.01,
                   job_id='', y0_pop=None, nrep=1):

    stoch_fn = SSAModel(base, vol, nrep=nrep, y0_pop=y0_pop)
    trajectories = stk.stochkit(stoch_fn,
                                job_id=base.model.getOption('name') +
                                str(job_id), t=t,
                                number_of_trajectories=ntraj,
                                increment=increment)

    ts = trajectories[0][:,0]
    traj = np.array(trajectories)[:,:,1:]

    # Restructure trajectory array to recreate (replicate, time, state)
    # indexing scheme
    traj = traj.reshape((ntraj, len(ts), nrep, base.NEQ)) # Expand
    traj = traj.swapaxes(2,1) # Bring reps and traj to front
    traj = traj.reshape((nrep*ntraj,len(ts), base.NEQ)) # Collapse to single axis
    traj = traj/vol # return concentrations

    return ts, traj




if __name__ == "__main__":

    base_control = create_class()
    base_control.limitCycle()

    # Calculate stochastic trajectories
    vol = 50000
    ts, traj = simulate_stoch(base_control, vol,
                              t=10*base_control.y0[-1], ntraj=10, nrep=1,
                              increment=base_control.y0[-1]/50)

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
