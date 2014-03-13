import numpy as np
import casadi as cs

# Model Constants
paramset = [9, 1, 0.56, 0.01, 8, 0.12, 0.3, 2, 0.05, 0.24, 0.02, 0.12,
            3.6, 2.16, 3, 0.75, 0.24, 0.06, 0.45, 0.06, 0.12, 0.09,
            0.003, 0.09]

y0in = [1.52385116,  1.11814894,  0.91371752,  0.27752431,  0.27487349,
        0.77658893,  1.05254706, 23.84237209]





NEQ = 7
NP = 24

def model():
    # Variable Assignments
    y1 = cs.ssym('y1')
    y2 = cs.ssym('y2')
    y3 = cs.ssym('y3')
    y4 = cs.ssym('y4')
    y5 = cs.ssym('y5')
    y6 = cs.ssym('y6')
    y7 = cs.ssym('y7')

    y = cs.vertcat([y1, y2, y3, y4, y5, y6, y7])
    
    # Parameter Assignments
    v1b = cs.ssym("v1b")
    k1b = cs.ssym("k1b")
    k1i = cs.ssym("k1i")
    c   = cs.ssym("c")
    p   = cs.ssym("p")
    k1d = cs.ssym("k1d")
    k2b = cs.ssym("k2b")
    q   = cs.ssym("q")
    k2d = cs.ssym("k2d")
    k2t = cs.ssym("k2t")
    k3t = cs.ssym("k3t")
    k3d = cs.ssym("k3d")
    v4b = cs.ssym("v4b")
    k4b = cs.ssym("k4b")
    r   = cs.ssym("r")
    k4d = cs.ssym("k4d")
    k5b = cs.ssym("k5b")
    k5d = cs.ssym("k5d")
    k5t = cs.ssym("k5t")
    k6t = cs.ssym("k6t")
    k6d = cs.ssym("k6d")
    k6a = cs.ssym("k6a")
    k7a = cs.ssym("k7a")
    k7d = cs.ssym("k7d")
        
    symparamset=cs.vertcat([v1b, k1b, k1i, c, p, k1d, k2b, q, k2d, k2t,
                            k3t, k3d, v4b, k4b, r, k4d, k5b, k5d, k5t,
                            k6t, k6d, k6a, k7a, k7d])
        
    # Time Variable
    t = cs.ssym("t")
    
    
    ode = [[]]*NEQ

    f1 = v1b*(y7 + c)/(k1b*(1 + (y3/k1i)**p) + y7 + c)
    f2 = v4b*y3**r/(k4b**r + y3**r) # Sabine had k4b**r, but Geier didn't

    ode[0] = f1 - k1d*y1
    ode[1] = k2b*y1**q - k2d*y2 - k2t*y2 + k3t*y3
    ode[2] = k2t*y2 - k3t*y3 - k3d*y3
    ode[3] = f2 - k4d*y4
    ode[4] = k5b*y4 - k5d*y5 - k5t*y5 + k6t*y6
    ode[5] = k5t*y5 - k6t*y6 - k6d*y6 + k7a*y7 - k6a*y6
    ode[6] = k6a*y6 - k7a*y7 - k7d*y7
    

    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Sabine Model (from stephanie)")
    
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
