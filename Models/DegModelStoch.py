import numpy as np

# John's files for stochasic evaluation
import CommonFiles.stochkit_resources as stk
import CommonFiles.modelbuilder as mb

# Deterministic model
from CommonFiles.Models.degModelFinal import create_class


def SSAModel(base, vol):
    y0_pop = (vol*base.y0[:-1]).astype(int)

    stoch_fn = stk.StochKitModel(
        name=base.model.getOption('name') + '_stoch')

    SSA_builder = mb.SSA_builder(base.ylabels, base.plabels, y0_pop,
                                 base.paramset, stoch_fn, vol)

    # ADD REACTIONS
    # =============
    
    # Per mRNA
    SSA_builder.SSA_MM_P('per mRNA repression', 'vtp', km=['knp'],
                         Prod=['p'], Rep=['C1P', 'C2P'], P=3)
    SSA_builder.SSA_MM('per mRNA degradation', 'vdp', km=['kdp'],
                       Rct=['p'])

    # Cry1 mRNA
    SSA_builder.SSA_MM_P('c1 mRNA repression', 'vtc1', km=['knc1'],
                         Prod=['c1'], Rep=['C1P', 'C2P'], P=3)
    SSA_builder.SSA_MM('c1 mRNA degradation', 'vdc1', km=['kdc1'],
                       Rct=['c1'])

    # Cry2 mRNA (shares cry1 km constants)
    SSA_builder.SSA_MM_P('c2 mRNA repression', 'vtc2', km=['knc1'],
                         Prod=['c2'], Rep=['C1P', 'C2P'], P=3)
    SSA_builder.SSA_MM('c2 mRNA degradation', 'vdc2', km=['kdc1'],
                       Rct=['c2'])

    # Free protein creation and degradation
    SSA_builder.SSA_MA_tln('PER translation', 'P', 'ktxnp', 'p')
    SSA_builder.SSA_MA_tln('CRY1 translation', 'C1', '1', 'c1')
    SSA_builder.SSA_MA_tln('CRY2 translation', 'C2', '1', 'c2')

    SSA_builder.SSA_MM('PER degradation', 'vdP', km=['kdP'], Rct=['P'])
    SSA_builder.SSA_MM('C1 degradation', 'vdC1', km=['kdC1'],
                       Rct=['C1'])
    SSA_builder.SSA_MM('C2 degradation', 'vdC2', km=['kdC1'],
                       Rct=['C2'])

    #CRY1 CRY2 complexing
    SSA_builder.SSA_MA_complex('CRY1-P complex', 'C1', 'P', 'C1P',
                               'vaC1P', 'vdC1P')
    SSA_builder.SSA_MA_complex('CRY2-P complex', 'C2', 'P', 'C2P',
                               'vaC1P', 'vdC1P')
    SSA_builder.SSA_MM('C1P degradation', 'vdCn', km=['kdCn'],
                       Rct=['C1P', 'C2P'])
    SSA_builder.SSA_MM_C2('C2P degradation', 'vdCn', km=['kdCn'],
                          Rct=['C2P', 'C1P'], M='MC2n')
    

    return stoch_fn


def simulate_stoch(base, vol, t=None, traj=100, increment=0.01,
                   job_id=''):

    stoch_fn = SSAModel(base, vol)
    trajectories = stk.stochkit(stoch_fn, job_id='deg_model' +
                                str(job_id), t=t,
                                number_of_trajectories=traj,
                                increment=increment)

    ts = trajectories[0][:,0]
    traj = np.array(trajectories)[:,:,1:]/vol
    return ts, traj


if __name__ == "__main__":

    base_control = create_class()

    vol = 500

    # Estimate decay parameter from stochastic trajectories
    ts, traj = simulate_stoch(base_control, vol,
                              t=5*base_control.y0[-1], traj=100,
                              increment=base_control.y0[-1]/100)
    from CommonFiles.DecayingSinusoid import DecayingSinusoid
    trans = len(ts)/4
    master = DecayingSinusoid(base_control._t_to_phi(ts[trans:]),
                              traj.mean(0)[trans:,1], max_degree=0,
                              decay_units='1/rad').run()
    phase_diff = master.averaged_params['decay'].value


    # Create amplitude model to capture diffusive effect
    from CommonFiles.Amplitude import (Amplitude,
                                       gaussian_phase_distribution)
    amp = Amplitude(base_control.model, base_control.paramset,
                    base_control.y0)
    population = gaussian_phase_distribution(0, 0, phase_diff)
    amp.phase_distribution = population
    amp._init_amp_class()
    x_bar = amp.x_bar(amp._t_to_phi(ts))


    # Plot deterministic and stochastic trajectories
    import matplotlib.pyplot as plt
    from CommonFiles.PlotOptions import PlotOptions, layout_pad
    PlotOptions(uselatex=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines_det = ax.plot(amp._t_to_phi(ts), x_bar[:,0])
    # ax.set_color_cycle(None)
    lines_sto = ax.plot(amp._t_to_phi(ts), traj.mean(0)[:,0], '--')
    ax.legend([lines_det[0], lines_sto[0]],
              ['Deterministic', 'Stochastic'])

    pi_ticks = np.arange(0,14,2)
    ax.set_xticks(pi_ticks*np.pi)
    ax.set_xticklabels(['0'] +
                        [r'$' + str(x) + r'\pi$' for x in pi_ticks[1:]])

    fig.tight_layout(**layout_pad)

    plt.show()
