import numpy as np

# John's files for stochasic evaluation
import CommonFiles.stochkit_resources as stk
import CommonFiles.modelbuilder as mb

# Deterministic model
from CommonFiles.Models.degModelFinal import create_class


def SSAModel(base, vol, nrep=1, y0_pop=None):
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

    for index in xrange(nrep):
        si = '_' + str(index)
    
        # Per mRNA
        bd.SSA_MM_P('per mRNA repression'+si, 'vtp'+si, km=['knp'+si],
                    Prod=['p'+si], Rep=['C1P'+si, 'C2P'+si], P=3)
        bd.SSA_MM('per mRNA degradation'+si, 'vdp'+si, km=['kdp'+si],
                  Rct=['p'+si])

        # Cry1 mRNA
        bd.SSA_MM_P('c1 mRNA repression'+si, 'vtc1'+si, km=['knc1'+si],
                    Prod=['c1'+si], Rep=['C1P'+si, 'C2P'+si], P=3)
        bd.SSA_MM('c1 mRNA degradation'+si, 'vdc1'+si, km=['kdc1'+si],
                  Rct=['c1'+si])

        # Cry2 mRNA (shares cry1 km constants)
        bd.SSA_MM_P('c2 mRNA repression'+si, 'vtc2'+si, km=['knc1'+si],
                    Prod=['c2'+si], Rep=['C1P'+si, 'C2P'+si], P=3)
        bd.SSA_MM('c2 mRNA degradation'+si, 'vdc2'+si, km=['kdc1'+si],
                  Rct=['c2'+si])

        # Free protein creation and degradation
        bd.SSA_MA_tln('PER translation'+si, 'P'+si, 'ktxnp'+si, 'p'+si)
        bd.SSA_MA_tln('CRY1 translation'+si, 'C1'+si, '1', 'c1'+si)
        bd.SSA_MA_tln('CRY2 translation'+si, 'C2'+si, '1', 'c2'+si)

        bd.SSA_MM('PER degradation'+si, 'vdP'+si, km=['kdP'+si],
                  Rct=['P'+si])
        bd.SSA_MM('C1 degradation'+si, 'vdC1'+si, km=['kdC1'+si],
                  Rct=['C1'+si])
        bd.SSA_MM('C2 degradation'+si, 'vdC2'+si, km=['kdC1'+si],
                  Rct=['C2'+si])

        #CRY1 CRY2 complexing
        bd.SSA_MA_complex('CRY1-P complex'+si, 'C1'+si, 'P'+si, 'C1P'+si,
                          'vaC1P'+si, 'vdC1P'+si)
        bd.SSA_MA_complex('CRY2-P complex'+si, 'C2'+si, 'P'+si, 'C2P'+si,
                          'vaC1P'+si, 'vdC1P'+si)
        bd.SSA_MM('C1P degradation'+si, 'vdCn'+si, km=['kdCn'+si],
                  Rct=['C1P'+si, 'C2P'+si])
        bd.SSA_MM_C2('C2P degradation'+si, 'vdCn'+si, km=['kdCn'+si],
                     Rct=['C2P'+si, 'C1P'+si], M='MC2n'+si)
    

    return stoch_fn


def simulate_stoch(base, vol, t=None, ntraj=1, increment=0.01,
                   job_id='', y0_pop=None, nrep=1):

    stoch_fn = SSAModel(base, vol, nrep=nrep, y0_pop=y0_pop)
    trajectories = stk.stochkit(stoch_fn, job_id='deg_model' +
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
    vol = 250
    ts, traj = simulate_stoch(base_control, vol,
                              t=10*base_control.y0[-1], ntraj=1000, nrep=1,
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
