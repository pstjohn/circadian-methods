# Model of a hysteresis-driven negative-feedback oscillator, taken from
# B. Novak and J. J. Tyson, "Design principles of biochemical
# oscillators," Nat. Rev. Mol. Cell Biol., vol. 9, no. 12, pp. 981-91,
# Dec. 2008.
# Figure 3, equation 8.


# This model has a very short period (~3), so routines should be run
# with a lower transient time

import casadi as cs

# Model Constants
paramset = [4., 20., 1., 0.005, 0.05, 0.1]
y0in = [ 0.65609071,  0.85088331,  3.04067965]

NEQ = 2
NP = 6

def model():
    # Variable Assignments
    X = cs.ssym("X")
    Y = cs.ssym("Y")

    y = cs.vertcat([X,Y])
    
    # Parameter Assignments
    P  = cs.ssym("P")
    kt = cs.ssym("kt")
    kd = cs.ssym("kd")
    a0 = cs.ssym("a0")
    a1 = cs.ssym("a1")
    a2 = cs.ssym("a2")
        
    symparamset = cs.vertcat([P, kt, kd, a0, a1, a2])
    # Time Variable
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
    from .. import pBase
    return pBase(model(), paramset, y0in)


