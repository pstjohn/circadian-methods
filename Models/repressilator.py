import casadi as cs

# Model Constants
paramset = [200., 0., 2., .2]

NEQ = 6
NP = 4

def model():
    #======================================================================
    # Variable Assignments
    #======================================================================
    m1 = cs.ssym("m1")
    p1 = cs.ssym("p1")
    m2 = cs.ssym("m2")
    p2 = cs.ssym("p2")
    m3 = cs.ssym("m3")
    p3 = cs.ssym("p3")

    y = cs.vertcat([m1, p1, m2, p2, m3, p3])

    #======================================================================
    # Parameter Assignments
    #======================================================================
    alpha  = cs.ssym("alpha")
    alpha0 = cs.ssym("alpha0")
    n      = cs.ssym("n")
    beta   = cs.ssym("beta")
        
    symparamset = cs.vertcat([alpha, alpha0, n, beta])

    # Time Variable
    t = cs.ssym("t")

    ode = [[]]*NEQ
    ode[0] = alpha0 + alpha/(1+p3**n) - m1
    ode[1] = beta*(m1-p1)
    ode[2] = alpha0 + alpha/(1+p1**n) - m2
    ode[3] = beta*(m2-p2)
    ode[4] = alpha0 + alpha/(1+p2**n) - m3
    ode[5] = beta*(m3-p3)
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Tyson 2 State Model")
    
    return fn

def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)


