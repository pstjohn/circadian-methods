import casadi as cs

# Define Constants

# Sensitivity Constants
ABSTOL = 1e-11
RELTOL = 1e-9
MAXNUMSTEPS = 40000
SENSMETHOD = "staggered"

# Model Constants
NEQ = 16
NP = 52

paramset = [0.4 , 0.2 , 0.4  , 0.2  , 0.4  , 0.2  , 0.5  , 0.1  , 0.7 ,
            0.6 , 2.2 , 0.01 , 0.01 , 0.01 , 0.01 , 0.12 , 0.3  , 0.1 ,
            0.1 , 0.4 , 0.4  , 0.31 , 0.12 , 1.6  , 0.6  , 2    , 4   ,
            0.5 , 0.6 , 0.4  , 0.4  , 0.1  , 0.1  , 0.3  , 0.1  , 0.5 ,
            0.4 , 0.2 , 0.1  ,  0.5  , 0.6  , 0.7  , 0.8  , 0.7 ,
            0.7 , 0.7 , 0.8  , 1    , 1.1  , 1    , 1.1  , 1.5]

y0in = [4.59407105,   2.86451577,   7.48884079,   1.05945104,
        8.08220288,   0.05748281,   0.70488658,   4.56530925,
        1.57704986,   0.24563968,   0.21626293,   2.01114509,
        0.92338126,   0.88380173,   0.31640511,   0.57828811,  23.80801488]
            
def model():

    # Variable Assignments
    MP   = cs.ssym("MP")
    MC   = cs.ssym("MC")
    MB   = cs.ssym("MB")
    PC   = cs.ssym("PC")
    CC   = cs.ssym("CC")
    PCP  = cs.ssym("PCP")
    CCP  = cs.ssym("CCP")
    PCC  = cs.ssym("PCC")
    PCN  = cs.ssym("PCN")
    PCCP = cs.ssym("PCCP")
    PCNP = cs.ssym("PCNP")
    BC   = cs.ssym("BC")
    BCP  = cs.ssym("BCP")
    BN   = cs.ssym("BN")
    BNP  = cs.ssym("BNP")
    IN   = cs.ssym("IN")
    
    y = [MP,MC,MB,PC,CC,PCP,CCP,PCC,PCN,PCCP,PCNP,BC,BCP,BN,BNP,IN]
    
    # Parameter Assignments
    k1    = cs.ssym("k1")
    k2    = cs.ssym("k2")
    k3    = cs.ssym("k3")
    k4    = cs.ssym("k4")
    k5    = cs.ssym("k5")
    k6    = cs.ssym("k6")
    k7    = cs.ssym("k7")
    k8    = cs.ssym("k8")
    KAP   = cs.ssym("KAP")
    KAC   = cs.ssym("KAC")
    KIB   = cs.ssym("KIB")
    kdmb  = cs.ssym("kdmb")
    kdmc  = cs.ssym("kdmc")
    kdmp  = cs.ssym("kdmp")
    kdn   = cs.ssym("kdn")
    kdnc  = cs.ssym("kdnc")
    Kd    = cs.ssym("Kd")
    Kdp   = cs.ssym("Kdp")
    Kp    = cs.ssym("Kp")
    KmB   = cs.ssym("KmB")
    KmC   = cs.ssym("KmC")
    KmP   = cs.ssym("KmP")
    ksB   = cs.ssym("ksB")
    ksC   = cs.ssym("ksC")
    ksP   = cs.ssym("ksP")
    m     = cs.ssym("m")
    n     = cs.ssym("n")
    V1B   = cs.ssym("V1B")
    V1C   = cs.ssym("V1C")
    V1P   = cs.ssym("V1P")
    V1PC  = cs.ssym("V1PC")
    V2B   = cs.ssym("V2B")
    V2C   = cs.ssym("V2C")
    V2P   = cs.ssym("V2P")
    V2PC  = cs.ssym("V2PC")
    V3B   = cs.ssym("V3B")
    V3PC  = cs.ssym("V3PC")
    V4B   = cs.ssym("V4B")
    V4PC  = cs.ssym("V4PC")
    vdBC  = cs.ssym("vdBC")
    vdBN  = cs.ssym("vdBN")
    vdCC  = cs.ssym("vdCC")
    vdIN  = cs.ssym("vdIN")
    vdPC  = cs.ssym("vdPC")
    vdPCC = cs.ssym("vdPCC")
    vdPCN = cs.ssym("vdPCN")
    vmB   = cs.ssym("vmB")
    vmC   = cs.ssym("vmC")
    vmP   = cs.ssym("vmP")
    vsB   = cs.ssym("vsB")
    vsC   = cs.ssym("vsC")
    vsP   = cs.ssym("vsP")
    
    p = [k1, k2, k3, k4, k5, k6, k7, k8, KAP, KAC, KIB, kdmb, kdmc, kdmp, kdn, kdnc, Kd, Kdp, Kp, KmB, KmC, KmP, ksB, ksC, ksP, m, n, V1B, V1C, V1P, V1PC, V2B, V2C, V2P, V2PC, V3B, V3PC, V4B, V4PC, vdBC, vdBN, vdCC, vdIN, vdPC, vdPCC, vdPCN, vmB, vmC, vmP, vsB, vsC, vsP]
    
    # Time Variable
    t = cs.ssym("t")
    
    ode = [[]]*NEQ
    #    /*  mRNA of per */
    ode[0] = (vsP*pow(BN,n)/(pow(KAP,n)  +  pow(BN,n))  -  vmP*MP/(KmP  +  MP)  -  kdmp*MP)
    
    #    /*  mRNA of cry */
    ode[1] = (vsC*pow(BN,n)/(pow(KAC,n)  +  pow(BN,n))  -  vmC*MC/(KmC  +  MC)  -  kdmc*MC)
    
    #    /*  mRNA of BMAL1  */
    ode[2] = (vsB*pow(KIB,m)/(pow(KIB,m) +  pow(BN,m))  -  vmB*MB/(KmB  +  MB)  -  kdmb *MB)
    
    #    /*  protein PER cytosol */
    ode[3] = (ksP*MP  -  V1P*PC/(Kp  +  PC)  +  V2P*PCP/(Kdp  +  PCP)  +  k4*PCC  -  k3*PC*CC  -  kdn*PC)
    
    #    /*  protein CRY cytosol */
    ode[4] = (ksC*MC - V1C*CC/(Kp + CC) + V2C*CCP/(Kdp + CCP) + k4*PCC - k3*PC*CC - kdnc*CC)
    
    #    /*  phosphorylated PER cytosol */
    ode[5] = (V1P*PC/(Kp + PC) - V2P*PCP/(Kdp + PCP) - vdPC*PCP/(Kdp + PCP) - kdn*PCP)
    
    #    /*  phosphorylated CRY cytosol  */
    ode[6] = (V1C*CC/(Kp + CC) - V2C*CCP/(Kdp + CCP) - vdCC*CCP/(Kd + CCP) - kdn*CCP)
    
    #    /*  PER:CRY complex cytosol */
    ode[7] = ( - V1PC*PCC/(Kp + PCC) + V2PC*PCCP/(Kdp + PCCP) - k4*PCC + k3*PC*CC + k2*PCN - k1*PCC - kdn*PCC)
    
    #    /* PER:CRY complex nucleus */
    ode[8] = ( - V3PC*PCN/(Kp + PCN) + V4PC*PCNP/(Kdp + PCNP) - k2*PCN + k1*PCC - k7*BN*PCN + k8*IN - kdn*PCN)
    
    #    /*  phopshorylated [PER:CRY)c cytosol */
    ode[9] = (V1PC*PCC/(Kp + PCC) - V2PC*PCCP/(Kdp + PCCP) - vdPCC*PCCP/(Kd + PCCP) - kdn*PCCP)
    
    #    /*  phosphorylated [PER:CRY)n */
    ode[10] = (V3PC*PCN/(Kp + PCN) - V4PC*PCNP/(Kdp + PCNP) - vdPCN*PCNP/(Kd + PCNP) - kdn*PCNP)
    
    #    /*  protein BMAL1 cytosol  */
    ode[11] = (ksB*MB - V1B*BC/(Kp + BC) + V2B*BCP/(Kdp + BCP) - k5*BC + k6*BN - kdn*BC)
    
    #    /* phosphorylated BMAL1 cytosol */
    ode[12] = (V1B*BC/(Kp + BC) - V2B*BCP/(Kdp + BCP) - vdBC*BCP/(Kd + BCP) - kdn*BCP)
    
    #    /*  protein BMAL1 nucleus */
    ode[13] = ( - V3B*BN/(Kp + BN) + V4B*BNP/(Kdp + BNP) + k5*BC - k6*BN - k7*BN*PCN  +  k8*IN - kdn*BN)
    
    #    /*  phosphorylatd BMAL1 nucleus */
    ode[14] = (V3B*BN/(Kp + BN) - V4B*BNP/(Kdp + BNP) - vdBN*BNP/(Kd + BNP) - kdn*BNP)
    
    #    /*  inactive complex between [PER:CRY)n abd [CLOCK:BMAL1)n */
    ode[15] = ( - k8*IN + k7*BN*PCN - vdIN*IN/(Kd + IN) - kdn*IN)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=cs.vertcat(y), p=cs.vertcat(p)),
                       cs.daeOut(ode=cs.vertcat(ode)))
    fn.setOption("name","Leloup 16")
    
    return fn

def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)
