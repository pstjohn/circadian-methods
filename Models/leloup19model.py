import casadi as cs
import numpy  as np


# Define Constants

# Sensitivity Constants
ABSTOL = 1e-11
RELTOL = 1e-9
MAXNUMSTEPS = 40000
SENSMETHOD = "staggered"

# Model Constants
NEQ = 19
NP = 95

paramset = [0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 1., 0.2, 0.8, 0.4, 0.6, 0.6,
            0.6, 1., 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 1.006, 1.006, 1.006, 1.006, 1.006,
            1.006, 0.3, 0.4, 0.4, 0.4, 1.2, 3.2, 0.32, 1.7, 2, 2, 2, 2,
            9.6, 2.4, 2.4, 0.6, 1.2, 0.2, 0.2, 0.2, 0.2, 1.4, 1.4, 0.4,
            3.4, 1.4, 1.4, 1.4, 3, 3, 1.6, 4.4, 0.8, 2.2, 2, 1.3, 1.6,
            2.4, 2.2, 1.8, 1.6]

paramset = np.array(paramset)
            
            
y0in = [4.03450825010985, 4.620722293666089, 8.431430939870038,
        0.02396802822535259, 328.0101675496989, 0.013854273648744166,
        0.7613257978733196, 3.997196870801472, 1.0959033787155004,
        5.382524712824021, 0.6371140788559054, 3.291255222410913,
        0.14046433295676652, 1.6797104018037374, 0.08987919355173075,
        1.6700933042040345, 2.6693533136876724, 1.5371240477539911,
        1.1301782623436725, 24.041880014967646]


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
    MR   = cs.ssym("MR")
    RC   = cs.ssym("RC")
    RN   = cs.ssym("RN")
    
    y = cs.vertcat([MP, MC, MB, PC, CC, PCP, CCP, PCC, PCN, PCCP, PCNP,
                    BC, BCP, BN, BNP, IN, MR, RC, RN])
    
    # Parameter Assignments
    k1       = cs.ssym('k1')
    k2       = cs.ssym('k2')
    k3       = cs.ssym('k3')
    k4       = cs.ssym('k4')
    k5       = cs.ssym('k5')
    k6       = cs.ssym('k6')
    k7       = cs.ssym('k7')
    k8       = cs.ssym('k8')
    k9       = cs.ssym('k9')
    k10      = cs.ssym('k10')
    KAP      = cs.ssym('KAP')
    KAC      = cs.ssym('KAC')
    KAR      = cs.ssym('KAR')
    KIB      = cs.ssym('KIB')
    kdmb     = cs.ssym('kdmb')
    kdmc     = cs.ssym('kdmc')
    kdmp     = cs.ssym('kdmp')
    kdmr     = cs.ssym('kdmr')
    kdn_c    = cs.ssym('kdn_c')
    kdn_p    = cs.ssym('kdn_p')
    kdn_pp   = cs.ssym('kdn_pp')
    kdn_cp   = cs.ssym('kdn_cp')
    kdn_pcc  = cs.ssym('kdn_pcc')
    kdn_pcn  = cs.ssym('kdn_pcn')
    kdn_pccp = cs.ssym('kdn_pccp')
    kdn_pcnp = cs.ssym('kdn_pcnp')
    kdn_bc   = cs.ssym('kdn_bc')
    kdn_bcp  = cs.ssym('kdn_bcp')
    kdn_bn   = cs.ssym('kdn_bn')
    kdn_bnp  = cs.ssym('kdn_bnp')
    kdn_in   = cs.ssym('kdn_in')
    kdn_rc   = cs.ssym('kdn_rc')
    kdn_rn   = cs.ssym('kdn_rn')
    Kd_p     = cs.ssym('Kd_p')
    Kd_c     = cs.ssym('Kd_c')
    Kd_pcc   = cs.ssym('Kd_pcc')
    Kd_pcn   = cs.ssym('Kd_pcn')
    Kd_bc    = cs.ssym('Kd_bc')
    Kd_bn    = cs.ssym('Kd_bn')
    Kd_in    = cs.ssym('Kd_in')
    Kd_rc    = cs.ssym('Kd_rc')
    Kd_rn    = cs.ssym('Kd_rn')
    Kdp_p    = cs.ssym('Kdp_p')
    Kdp_c    = cs.ssym('Kdp_c')
    Kdp_pcc  = cs.ssym('Kdp_pcc')
    Kdp_pcn  = cs.ssym('Kdp_pcn')
    Kdp_bc   = cs.ssym('Kdp_bc')
    Kdp_bn   = cs.ssym('Kdp_bn')
    Kp_p     = cs.ssym('Kp_p')
    Kp_c     = cs.ssym('Kp_c')
    Kp_pcc   = cs.ssym('Kp_pcc')
    Kp_pcn   = cs.ssym('Kp_pcn')
    Kp_bc    = cs.ssym('Kp_bc')
    Kp_bn    = cs.ssym('Kp_bn')
    KmP      = cs.ssym('KmP')
    KmC      = cs.ssym('KmC')
    KmB      = cs.ssym('KmB')
    KmR      = cs.ssym('KmR')
    ksP      = cs.ssym('ksP')
    ksC      = cs.ssym('ksC')
    ksB      = cs.ssym('ksB')
    ksR      = cs.ssym('ksR')
    h        = cs.ssym('h')
    m        = cs.ssym('m')
    n_p      = cs.ssym('n_p')
    n_c      = cs.ssym('n_c')
    V1P      = cs.ssym('V1P')
    V1PC     = cs.ssym('V1PC')
    V3PC     = cs.ssym('V3PC')
    V2P      = cs.ssym('V2P')
    V1C      = cs.ssym('V1C')
    V2C      = cs.ssym('V2C')
    V2B      = cs.ssym('V2B')
    V2PC     = cs.ssym('V2PC')
    V4PC     = cs.ssym('V4PC')
    V1B      = cs.ssym('V1B')
    V3B      = cs.ssym('V3B')
    V4B      = cs.ssym('V4B')
    vdPC     = cs.ssym('vdPC')
    vdCC     = cs.ssym('vdCC')
    vdPCC    = cs.ssym('vdPCC')
    vdPCN    = cs.ssym('vdPCN')
    vdBC     = cs.ssym('vdBC')
    vdBN     = cs.ssym('vdBN')
    vdIN     = cs.ssym('vdIN')
    vdRC     = cs.ssym('vdRC')
    vdRN     = cs.ssym('vdRN')
    vmP      = cs.ssym('vmP')
    vmC      = cs.ssym('vmC')
    vmB      = cs.ssym('vmB')
    vmR      = cs.ssym('vmR')
    vsP      = cs.ssym('vsP')
    vsC      = cs.ssym('vsC')
    vsB      = cs.ssym('vsB')
    vsR      = cs.ssym('vsR')

    
    param = cs.vertcat([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, KAP,
                        KAC, KAR, KIB, kdmb, kdmc, kdmp, kdmr, kdn_c,
                        kdn_p, kdn_pp, kdn_cp, kdn_pcc, kdn_pcn,
                        kdn_pccp, kdn_pcnp, kdn_bc, kdn_bcp, kdn_bn,
                        kdn_bnp, kdn_in, kdn_rc, kdn_rn, Kd_p, Kd_c,
                        Kd_pcc, Kd_pcn, Kd_bc, Kd_bn, Kd_in, Kd_rc,
                        Kd_rn, Kdp_p, Kdp_c, Kdp_pcc, Kdp_pcn, Kdp_bc,
                        Kdp_bn, Kp_p, Kp_c, Kp_pcc, Kp_pcn, Kp_bc,
                        Kp_bn, KmP, KmC, KmB, KmR, ksP, ksC, ksB, ksR,
                        h, m, n_p, n_c, V1P, V1PC, V3PC, V2P, V1C, V2C,
                        V2B, V2PC, V4PC, V1B, V3B, V4B, vdPC, vdCC,
                        vdPCC, vdPCN, vdBC, vdBN, vdIN, vdRC, vdRN, vmP,
                        vmC, vmB, vmR, vsP, vsC, vsB, vsR])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    ode = [[]]*NEQ
    #    /*  mRNA of per */
    ode[0] = (vsP * pow(BN,n_p) / (pow(KAP,n_p) + pow(BN,n_p)) - vmP*MP / (KmP + MP) - kdmp * MP)
    
    #    /*  mRNA of cry */
    ode[1] = (vsC * pow(BN,n_c) / (pow(KAC,n_c) + pow(BN,n_c)) - vmC*MC / (KmC + MC) - kdmc * MC)
    
    #    /*  mRNA of BMAL1  */
    ode[2] = (vsB * pow(KIB,m) / (pow(KIB,m)+ pow(RN,m)) - vmB * MB / (KmB + MB) - kdmb *MB)
    
    #    /*  protein PER cytosol */
    ode[3] = (ksP * MP-V1P * PC / (Kp_p+PC)+V2P * PCP / (Kdp_p+PCP)+k4 * PCC-k3 * PC * CC-kdn_p * PC)
    
    #    /*  protein CRY cytosol */
    ode[4] = (ksC * MC-V1C * CC / (Kp_c+CC)+V2C * CCP / (Kdp_c+CCP)+k4 * PCC-k3 * PC * CC-kdn_c * CC)
    
    #    /*  phosphorylated PER cytosol */
    ode[5] = (V1P * PC / (Kp_p+PC)-V2P * PCP / (Kdp_p+PCP)-vdPC * PCP / (Kd_p+PCP)-kdn_pp * PCP)
    
    #    /*  phosphorylated CRY cytosol  */
    ode[6] = (V1C * CC / (Kp_c+CC)-V2C * CCP / (Kdp_c+CCP)-vdCC * CCP / (Kd_c+CCP)-kdn_cp * CCP)
    
    #    /*  PER:CRY complex cytosol */
    ode[7] = (-V1PC * PCC / (Kp_pcc+PCC)+V2PC * PCCP / (Kdp_pcc+PCCP)-k4 * PCC+k3 * PC * CC+k2 * PCN-k1 * PCC-kdn_pcc * PCC)
    
    #    /* PER:CRY complex nucleus */
    ode[8] = (-V3PC * PCN / (Kp_pcn+PCN)+V4PC * PCNP / (Kdp_pcn+PCNP)-k2 * PCN+k1 * PCC-k7 * BN * PCN+k8 * IN-kdn_pcn * PCN)
    
    #    /*  phopshorylated [PER:CRY)c cytosol */
    ode[9] = (V1PC * PCC / (Kp_pcc+PCC)-V2PC * PCCP / (Kdp_pcc+PCCP)-vdPCC * PCCP / (Kd_pcc+PCCP)-kdn_pccp * PCCP)
    
    #    /*  phosphorylated [PER:CRY)n */
    ode[10] = (V3PC * PCN / (Kp_pcn+PCN)-V4PC * PCNP / (Kdp_pcn+PCNP)-vdPCN * PCNP / (Kd_pcn+PCNP)-kdn_pcnp * PCNP)
    
    #    /*  protein BMAL1 cytosol  */
    ode[11] = (ksB * MB-V1B * BC / (Kp_bc+BC)+V2B * BCP / (Kdp_bc+BCP)-k5 * BC+k6 * BN-kdn_bc * BC)
    
    #    /* phosphorylated BMAL1 cytosol */
    ode[12] = (V1B * BC / (Kp_bc+BC)-V2B * BCP / (Kdp_bc+BCP)-vdBC * BCP / (Kd_bc+BCP)-kdn_bcp * BCP)
    
    #    /*  protein BMAL1 nucleus */
    ode[13] = (-V3B * BN / (Kp_bn+BN)+V4B * BNP / (Kdp_bn+BNP)+k5 * BC-k6 * BN-k7 * BN * PCN + k8 * IN-kdn_bn * BN)
    
    #    /*  phosphorylatd BMAL1 nucleus */
    ode[14] = (V3B * BN / (Kp_bn+BN)-V4B * BNP / (Kdp_bn+BNP)-vdBN * BNP / (Kd_bn+BNP)-kdn_bnp * BNP)
    
    #    /*  inactive complex between [PER:CRY)n abd [CLOCK:BMAL1)n */
    ode[15] = (-k8 * IN+k7 * BN * PCN-vdIN * IN / (Kd_in+IN)-kdn_in * IN)

    ode[16] = vsR*BN**h/(KAR**h+BN**h)-vmR*MR/(KmR+MR)-kdmr*MR

    ode[17] = ksR*MR-k9*RC+k10*RN-vdRC*RC/(Kd_rc+RC)-kdn_rc*RC

    ode[18] = k9*RC-k10*RN-vdRN*RN/(Kd_rn+RN)-kdn_rn*RN


    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=param),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","Leloup 19")
    
    return fn


def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)
