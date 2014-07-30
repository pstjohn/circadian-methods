import numpy  as np
import casadi as cs

NEQ = 8
NP = 21

y0in = np.array([1.33593236, 1.79197778, 1.62225403, 1.62164561,
                 5.17596319, 3.96326956, 0.37392169, 0.1865519,
                 23.69999994])

paramset= np.array([  1.94599211e-01,   1.30641511e-01,   1.13505694e-01,
                      4.25460804e-01,   2.59493541e-01,   3.26290677e-01,
                      6.76063523e-01,   6.07937881e-01,   1.14786910e-02,
                      1.14911580e+00,   2.96993795e+00,   3.38240851e-02,
                      1.52309394e+00,   1.68558083e+00,   2.01721235e+00,
                      1.01201779e-01,   3.35780360e-01,   5.26256372e-02,
                      4.06306642e-02,   1.75478279e-03,   3.00000000e+00])


def model():
    """ trying non-reversible nuclear entry and fitting knockouts. """
    # Variable Assignments
    p   = cs.ssym("p")
    c1  = cs.ssym("c1")
    c2  = cs.ssym("c2")
    P   = cs.ssym("P")
    C1  = cs.ssym("C1")
    C2  = cs.ssym("C2")
    C1P = cs.ssym("C1P")
    C2P = cs.ssym("C2P")
    
    y = cs.vertcat([p, c1, c2, P, C1, C2, C1P, C2P])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    # Parameter Assignments
    vtp    = cs.ssym("vtp")
    vtc1   = cs.ssym("vtc1")
    vtc2   = cs.ssym("vtc2")
    knp    = cs.ssym("knp")
    knc1   = cs.ssym("knc1")
    vdp    = cs.ssym("vdp")
    vdc1   = cs.ssym("vdc1")
    vdc2   = cs.ssym("vdc2")
    kdp    = cs.ssym("kdp")
    kdc1   = cs.ssym("kdc1")
    vdP    = cs.ssym("vdP")
    vdC1   = cs.ssym("vdC1")
    vdC2   = cs.ssym("vdC2")
    vdC1n   = cs.ssym("vdC1n")
    vdC2n   = cs.ssym("vdC2n")
    kdP    = cs.ssym("kdP")
    kdC1   = cs.ssym("kdC1")
    kdCn   = cs.ssym("kdCn")
    vaC1P  = cs.ssym("vaC1P")
    vdC1P  = cs.ssym('vdC1P')
    ktxnp  = cs.ssym('ktxnp')


    param = cs.vertcat([vtp  , vtc1 , vtc2  , knp   , knc1   , vdp  ,
                        vdc1 , vdc2  , kdp  , kdc1 , vdP   , kdP   ,
                        vdC1   , vdC2 , kdC1 , vdC1n , vdC2n , kdCn ,
                        vaC1P , vdC1P , ktxnp])
    
    # Model Equations
    ode = [[]]*NEQ
    
    def txn(vmax,km,np,dact1,dact2):
        return vmax/(km + (dact1 + dact2)**np)
    
    def txl(mrna,kt):
        return kt*mrna
    
    def MMdeg(species,vmax,km):
        return -vmax*(species)/(km+species)
        
    def cxMMdeg(species1,species2,vmax,km):
        return -vmax*(species1)/(km + species1 + species2)
        
    def cx(s1,s2,cmplx,ka,kd):
        # positive for reacting species, negative for complex
        return -ka*s1*s2 + kd*cmplx
        
    # MRNA Species
    ode[0] = txn(vtp,knp,3,C1P,C2P)   + MMdeg(p,vdp,kdp)
    ode[1] = txn(vtc1,knc1,3,C1P,C2P) + MMdeg(c1,vdc1,kdc1)
    ode[2] = txn(vtc2,knc1,3,C1P,C2P) + MMdeg(c2,vdc2,kdc1)
    
    # Free Proteins
    ode[3] = txl(p,ktxnp) + MMdeg(P,vdP,kdP)    + cx(P,C1,C1P,vaC1P,vdC1P)\
                                                + cx(P,C2,C2P,vaC1P,vdC1P)
    ode[4] = txl(c1,1)    + MMdeg(C1,vdC1,kdC1) + cx(P,C1,C1P,vaC1P,vdC1P)
    ode[5] = txl(c2,1)    + MMdeg(C2,vdC2,kdC1) + cx(P,C2,C2P,vaC1P,vdC1P)

    # PER/CRY Cytoplasm Complexes
    ode[6] = -cx(P,C1,C1P,vaC1P,vdC1P) + cxMMdeg(C1P,C2P,vdC1n,kdCn)
    ode[7] = -cx(P,C2,C2P,vaC1P,vdC1P) + cxMMdeg(C2P,C1P,vdC2n,kdCn)

    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=param),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","degmodel")
    
    return fn

def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)
    
    
