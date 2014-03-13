import numpy as np
import casadi as cs

# Model Constants
paramset = [1,]
y0in = [ 2.00861986, -0.69267507,  6.66328686]


NEQ = 2

def model():

    # Time Variable
    t = cs.ssym("t")

    # Variable Assignments
    X = cs.ssym("X")
    Y = cs.ssym("Y")

    y = cs.vertcat([X,Y])
    
    # Parameter Assignments
    u  = cs.ssym("u")

    ode = [[]]*NEQ
    ode[0] = u*(X - (X**3/3.) - Y)
    ode[1] = X/u
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=cs.vertcat([u,])),
                       cs.daeOut(ode=ode))

    fn.setOption("name", "Van der Pol oscillator")
    
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
        print new.y0
