# A model of a simple mass-action chemical limit cycle oscillator,
# originally proposed in R. J. Field and R. M. Noyes, J. Chem. Phys.,
# 60, 1877 (1973)
#
# Stochastic version using Gillespie's algorithm is presented in his
# original paper, D. T. Gillespie, J. Phys. Chem. 81, 2340-2361 (1977).
#

import numpy as np
import casadi as cs


# Aux. parameters, according to Gillespie (p. 2358)
y1s = 500.
y2s = 1000.
y3s = 2000.
p1s = 2000.
p2s = 50000.

# [c1x1, c2, c3x2, c4, c5x3], according to Gillespie's paper
paramset = [
    p1s/y2s,
    p2s/(y1s*y2s),
    (p1s + p2s)/y1s,
    2*p1s/y1s**2,
    (p1s + p2s)/y3s]


y0in = [2866.2056016028037, 585.492592883311, 6874.38507145332,
        0.7033292578984436]

NEQ = 3
NP = 5

def model():
    """ Create an ODE casadi SXFunction """

    # State Variables
    Y1 = cs.ssym("Y1")
    Y2 = cs.ssym("Y2")
    Y3 = cs.ssym("Y3")

    y = cs.vertcat([Y1, Y2, Y3])

    # Parameters
    c1x1 = cs.ssym("c1x1")
    c2   = cs.ssym("c2")
    c3x2 = cs.ssym("c3x2")
    c4   = cs.ssym("c4")
    c5x3 = cs.ssym("c5x3")

    symparamset = cs.vertcat([c1x1, c2, c3x2, c4, c5x3])

    # Time
    t = cs.ssym("t")


    # ODES
    ode = [[]]*NEQ
    ode[0] = c1x1*Y2 - c2*Y1*Y2 + c3x2*Y1 - c4*Y1**2
    ode[1] = -c1x1*Y2 - c2*Y1*Y2 + c5x3*Y3
    ode[2] = c3x2*Y1 - c5x3*Y3
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name", "Oregonator")
    
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

    # ADD REACTIONS
    # =============
    #
    for index in xrange(nrep):
        si = '_' + str(index)

        bd.SSA_MA(name="Y1_production" + si,
                  reactants={bd.species_array[bd.ydict['Y2' + si]]:1},
                  products={bd.species_array[bd.ydict['Y1' + si]]:1},
                  rate=bd.param_array[bd.pdict['c1x1' + si]])

        bd.SSA_MA(name="Y1_Y2_deg" + si,
                  reactants={bd.species_array[bd.ydict['Y1' + si]]:1,
                             bd.species_array[bd.ydict['Y2' + si]]:1},
                  rate=bd.param_array[bd.pdict['c2' + si]])

        bd.SSA_MA(name="Y1_X2" + si,
                  reactants={bd.species_array[bd.ydict['Y1' + si]]:1},
                  products={bd.species_array[bd.ydict['Y1' + si]]:2,
                            bd.species_array[bd.ydict['Y3' + si]]:1},
                  rate=bd.param_array[bd.pdict['c3x2' + si]])

        bd.SSA_MA(name="Y1_deg" + si,
                  reactants={bd.species_array[bd.ydict['Y1' + si]]:2},
                  rate=bd.param_array[bd.pdict['c4' + si]])

        bd.SSA_MA(name="Y2_prod" + si,
                  reactants={bd.species_array[bd.ydict['Y3' + si]]:1},
                  products={bd.species_array[bd.ydict['Y2' + si]]:1},
                  rate=bd.param_array[bd.pdict['c5x3' + si]])

    return stoch_fn


def simulate_stoch(base, vol, t=None, ntraj=1, increment=0.01,
                   job_id='', y0_pop=None, nrep=1):

    stoch_fn = SSAModel(base, vol, nrep=nrep, y0_pop=y0_pop)
    trajectories = stk.stochkit(stoch_fn, job_id='Oregonator' +
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
    vol = 0.1
    ts, traj = simulate_stoch(base_control, vol,
                              t=5*base_control.y0[-1], ntraj=100,
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
