import casadi as cs
import numpy as np

# Define Constants

# Model Constants
paramset = [0.4, 0.12, 0.18, 0.12, 0.06, 12, 1, 1]
y0in = [2.12497159, 2.77656151, 1.09345771, 23.82250182]

NEQ = 3
NP = 8

def model():
    # Variable Assignments
    x = cs.ssym("x")
    y = cs.ssym("y")
    z = cs.ssym("z")

    symy = cs.vertcat([x,y,z])
    
    # Parameter Assignments
    b = cs.ssym("b")
    d1 = cs.ssym("d1")
    d2 = cs.ssym("d2")
    d3 = cs.ssym("d3")
    t1 = cs.ssym("t1")
    h = cs.ssym("h")
    V = cs.ssym("V")
    K = cs.ssym("K")

    symparamset = cs.vertcat([b, d1, d2, d3, t1, h, V, K])
    # Time Variable
    t = cs.ssym("t")
    
    
    ode = [[]]*NEQ
    
    #    /*  mRNA of per */
    ode[0] = V/(1 + (z/K)**h) - d1*x
    ode[1] = b*x - (d2 + t1)*y
    ode[2] = t1*y - d3*z

    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=symy, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Reischl Model")
    
    return fn

def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)


if __name__ == "__main__":
    from CommonFiles.pBase import pBase

    try: 
        new = pBase(model(), paramset, np.ones(NEQ+1))
        new.calcY0()
        print new.y0
    except Exception as ex:
        print ex


