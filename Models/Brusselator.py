# A model of a simple mass-action chemical limit cycle oscillator,
# originally proposed in Prigogine I., Lefever R. J. Chem. Phys. 1968;48.
#
# Stochastic version using Gillespie's algorithm is presented in his
# original paper, D. T. Gillespie, J. Phys. Chem. 81, 2340-2361 (1977).
#

import numpy as np
import casadi as cs


# [c1x1, c2x2, c3, c4], according to Gillespie's paper
paramset = [5000., 50., 0.00005, 5.]
y0in = [6719.395774605672, 322.9807399011612, 1.5381190783073013]

NEQ = 2
NP = 4

def model():
    """ Create an ODE casadi SXFunction """

    # State Variables
    Y1 = cs.ssym("Y1")
    Y2 = cs.ssym("Y2")

    y = cs.vertcat([Y1, Y2])

    # Parameters
    c1x1 = cs.ssym("c1x1")
    c2x2 = cs.ssym("c2x2")
    c3   = cs.ssym("c3")
    c4   = cs.ssym("c4")

    symparamset = cs.vertcat([c1x1, c2x2, c3, c4])

    # Time
    t = cs.ssym("t")


    # ODES
    ode = [[]]*NEQ
    ode[0] = c1x1 - c2x2*Y1 + (c3/2.)*(Y1**2)*Y2 - c4*Y1
    ode[1] = c2x2*Y1 - (c3/2.)*(Y1**2)*Y2
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name", "Brusselator")
    
    return fn

def create_class():
    from CommonFiles.pBase import pBase
    return pBase(model(), paramset, y0in)


# John's files for stochasic evaluation
import CommonFiles.stochkit_resources as stk
import CommonFiles.modelbuilder as mb

def SSAModel(base, vol):
    """ Abandoning, as stochkit (at least the python version) does not
    want to handle a tri-molecular reaction. Expansion into multiple
    events (i.e. oregonator) will be tried next """

    y0_pop = (vol*base.y0[:-1]).astype(int)

    stoch_fn = stk.StochKitModel(
        name=base.model.getOption('name') + '_stoch')

    bd = mb.SSA_builder(base.ylabels, base.plabels, y0_pop,
                        base.paramset, stoch_fn, vol)

    # ADD REACTIONS
    # =============
    rxn1 = stk.Reaction(name="Y1_production",
                        products={bd.species_array[bd.ydict['Y1']]:1},
                        massaction=True,
                        rate=bd.param_array[bd.pdict['c1x1']])
    bd.SSAmodel.addReaction(rxn1)

    rxn2 = stk.Reaction(name="Y2_from_Y1",
                        reactants={bd.species_array[bd.ydict['Y1']]:1},
                        products={bd.species_array[bd.ydict['Y2']]:1},
                        massaction=True,
                        rate=bd.param_array[bd.pdict['c2x2']])
    bd.SSAmodel.addReaction(rxn2)

    rxn3 = stk.Reaction(name="trimolecular",
                        reactants={bd.species_array[bd.ydict['Y1']]:2,
                                   bd.species_array[bd.ydict['Y2']]:1},
                        products={bd.species_array[bd.ydict['Y1']]:3},
                        massaction=True,
                        rate=bd.param_array[bd.pdict['c3']])
    bd.SSAmodel.addReaction(rxn3)

    rxn4 = stk.Reaction(name="Y2_deg",
                        reactants={bd.species_array[bd.ydict['Y2']]:1},
                        massaction=True,
                        rate=bd.param_array[bd.pdict['c4']])
    bd.SSAmodel.addReaction(rxn4)

    return stoch_fn


def simulate_stoch(base, vol, t=None, traj=100, increment=0.01,
                   job_id=''):

    stoch_fn = SSAModel(base, vol)
    trajectories = stk.stochkit(stoch_fn, job_id='brusselator' +
                                str(job_id), t=t,
                                number_of_trajectories=traj,
                                increment=increment)

    ts = trajectories[0][:,0]
    traj = np.array(trajectories)[:,:,1:]/vol
    return ts, traj



if __name__ == "__main__":

    base_control = create_class()

    # Calculate stochastic trajectories
    vol = 1
    ts, traj = simulate_stoch(base_control, vol,
                              t=5*base_control.y0[-1], traj=100,
                              increment=base_control.y0[-1]/100)





    # # Estimate decay parameter from stochastic trajectories
    # from CommonFiles.DecayingSinusoid import DecayingSinusoid
    # trans = len(ts)/4
    # master = DecayingSinusoid(base_control._t_to_phi(ts[trans:]),
    #                           traj.mean(0)[trans:,1],
    #                           max_degree=1).run()
    # phase_diff = -master.averaged_params['decay'].value


    # # Create amplitude model to capture diffusive effect
    # from CommonFiles.Amplitude import (Amplitude,
    #                                    gaussian_phase_distribution)
    # amp = Amplitude(base_control.model, base_control.paramset,
    #                 base_control.y0)
    # population = gaussian_phase_distribution(0, 0, phase_diff)
    # amp.phase_distribution = population
    # amp._init_amp_class()
    # x_bar = amp.x_bar(amp._t_to_phi(ts))


    # # Plot deterministic and stochastic trajectories
    # import matplotlib.pyplot as plt
    # from CommonFiles.PlotOptions import PlotOptions, layout_pad
    # PlotOptions(uselatex=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # lines_det = ax.plot(amp._t_to_phi(ts), x_bar[:,0])
    # ax.set_color_cycle(None)
    # lines_sto = ax.plot(amp._t_to_phi(ts), traj.mean(0)[:,0], '--')
    # ax.legend([lines_det[0], lines_sto[0]],
    #           ['Deterministic', 'Stochastic'])

    # pi_ticks = np.arange(0,14,2)
    # ax.set_xticks(pi_ticks*np.pi)
    # ax.set_xticklabels(['0'] +
    #                     [r'$' + str(x) + r'\pi$' for x in pi_ticks[1:]])

    # fig.tight_layout(**layout_pad)

    plt.show()
