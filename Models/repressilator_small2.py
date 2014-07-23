import casadi as cs

# Model Constants
y0in = [ 1.59152259,  1.14809173,  1.17365822,  3.66621715]
paramset = [4.0, 4.0, 4.0, 3.0]


NEQ = 3
NP = 2

def model():
    #======================================================================
    # Variable Assignments
    #======================================================================
    m1 = cs.ssym("m1")
    m2 = cs.ssym("m2")
    m3 = cs.ssym("m3")

    y = cs.vertcat([m1, m2, m3])

    #======================================================================
    # Parameter Assignments
    #======================================================================
    alpha1  = cs.ssym("alpha1")
    alpha2  = cs.ssym("alpha2")
    alpha3  = cs.ssym("alpha3")
    n      = cs.ssym("n")
        
    symparamset = cs.vertcat([alpha1, alpha2, alpha3, n])

    # Time Variable
    t = cs.ssym("t")

    ode = [[]]*NEQ
    ode[0] = alpha1/(1.+m2**n) - m1
    ode[1] = alpha2/(1.+m3**n) - m2
    ode[2] = alpha3/(1.+m1**n) - m3
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Small repressilator")
    
    return fn

def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)


