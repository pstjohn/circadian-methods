import casadi as cs

# Define Constants
NEQ = 12
NP = 51

y0in = [3.18905639,   3.8487871,   2.49418162,  13.5446698,
        0.29856141,   0.09180213,  12.62474256,   4.05666345,
        0.73696792,   0.46171796,   0.58005922,   0.44990306,
        7.32652564]

paramset = [0.3687, 2.2472, 0.2189, 0.5975, 0.7165, 0.5975*1.1444,
            1.0087, 0.9754, 0.9135, 1.9796, 1.4992, 0.6323, 0.0151,
            0.2447, 1.9004, 0.0603, 0.7254, 3.2261, 1.7679, 1.0821,
            0.0931, 1.1853, 0.8856, 0.3118, 0.0236, 1.2620, 1.6294,
            1.0136, 1.8596, 0.7780, 1.9296, 0.7348, 0.1052, 1.4786,
            0.0110, 9.3227, 1.3309, 0.4586, 1.4281, 6.9448, 0.9559,
            1.7986, 0.0151*1.1624, 1.1596, 0.6551, 0.0164, 0.9501,
            0.6327, 0.0954, 0.1338, 0.1907]

def model():

    # Variable Assignments
    MP   = cs.ssym("MP")   # mRNA of per
    MC   = cs.ssym("MC")   # mRNA of cry
    PC   = cs.ssym("PC")   # PER protein (cytosol)
    CC   = cs.ssym("CC")   # CRY protein (cytosol)
    PCC  = cs.ssym("PCC")  # PER:CRY complex (cytosol)
    PCN  = cs.ssym("PCN")  # PER:CRY complex (nucleus)
    BC   = cs.ssym("BC")   # protein BMAL1 (cytosol)
    BN   = cs.ssym("BN")   # [CLOCK:BMAL1]==nuclear BMAL1 protein
    BNac = cs.ssym("BNac") # Transcriptionally active (acetylated) BMAL1 complex
    MN   = cs.ssym("MN")   # mRNA of Nampt
    N    = cs.ssym("N")    # protein NAMPT
    NAD  = cs.ssym("NAD")  # cellular NAD levels
    
    y = cs.vertcat([MP, MC, PC, CC, PCC, PCN, BC, BN, BNac, MN, N, NAD])

    # Parameter Assignments
    n_n = 1; 
    n_p = 1; 
    n_c = 1; 
    r_n = 3; 
    r_p = 3; 
    r_c = 3; 
    
    V1PC     = cs.ssym("V1PC")
    KAN      = cs.ssym("KAN")
    V3PC     = cs.ssym("V3PC")
    k1       = cs.ssym("k1")
    vmN      = cs.ssym("vmN")
    k2       = cs.ssym("k2")
    Kp_pcc   = cs.ssym("Kp_pcc")
    Kp_bc    = cs.ssym("Kp_bc")
    KmN      = cs.ssym("KmN")
    Kp_c     = cs.ssym("Kp_c")
    vmP      = cs.ssym("vmP")
    k5       = cs.ssym("k5")
    k3       = cs.ssym("k3")
    V1B      = cs.ssym("V1B")
    vsn      = cs.ssym("vsn")
    V1C      = cs.ssym("V1C")
    Kp_pcn   = cs.ssym("Kp_pcn")
    Kac_bn   = cs.ssym("Kac_bn")
    Kd_bn    = cs.ssym("Kd_bn")
    V4B      = cs.ssym("V4B")
    Kdac_bn  = cs.ssym("Kdac_bn")
    MB0      = cs.ssym("MB0")
    KAP      = cs.ssym("KAP")
    KAC      = cs.ssym("KAC")
    vdBN     = cs.ssym("vdBN")
    V3B      = cs.ssym("V3B")
    ksN      = cs.ssym("ksN")
    sn       = cs.ssym("sn")
    vm_NAMPT = cs.ssym("vm_NAMPT")
    vm_NAD   = cs.ssym("vm_NAD")
    Km_NAMPT = cs.ssym("Km_NAMPT")
    Km_NAD   = cs.ssym("Km_NAD")
    v0p      = cs.ssym("v0p")
    v0c      = cs.ssym("v0c")
    v0n      = cs.ssym("v0n")
    vsP      = cs.ssym("vsP")
    vsC      = cs.ssym("vsC")
    kd       = cs.ssym("kd")
    k6       = cs.ssym("k6")
    ksB      = cs.ssym("ksB")
    ksP      = cs.ssym("ksP")
    ksC      = cs.ssym("ksC")
    k4       = cs.ssym("k4")
    Kp_p     = cs.ssym("Kp_p")
    vmC      = cs.ssym("vmC")
    KmP      = cs.ssym("KmP")
    KmC      = cs.ssym("KmC")
    V1P      = cs.ssym("V1P")
    Rp       = cs.ssym("Rp")
    Rc       = cs.ssym("Rc")
    Rn       = cs.ssym("Rn")

    symparamset = cs.vertcat(
        [V1PC, KAN, V3PC, k1, vmN, k2, Kp_pcc, Kp_bc, KmN, Kp_c, vmP,
         k5, k3, V1B, vsn, V1C, Kp_pcn, Kac_bn, Kd_bn, V4B, Kdac_bn,
         MB0, KAP, KAC, vdBN, V3B, ksN, sn, vm_NAMPT, vm_NAD, Km_NAMPT,
         Km_NAD, v0p, v0c, v0n, vsP, vsC, kd, k6, ksB, ksP, ksC, k4,
         Kp_p, vmC, KmP, KmC, V1P, Rp, Rc, Rn])

    # Time Variable
    t = cs.ssym("t")
    
    # Model Equations
    ode = [[]]*NEQ
    
    ode[0]  = v0p + vsP*BNac**n_p/(KAP**n_p*(1 + (PCN/Rp)**r_p) + BNac**n_p)-vmP*MP/(KmP+MP)-kd*MP       # Per
    ode[1]  = v0c + vsC*BNac**n_c/(KAC**n_c*(1 + (PCN/Rc)**r_c) + BNac**n_c) - vmC*MC/(KmC+MC)- kd*MC    # Cry
    ode[2]  = ksP*MP + k4*PCC - k3*PC*CC - V1P*PC/(Kp_p + PC) - kd*PC                                    # PER CYTOSOL
    ode[3]  = ksC*MC + k4*PCC - k3*PC*CC - V1C*CC/(Kp_c + CC) - kd*CC                                    # CRY CYTOSOL
    ode[4]  = (k3*PC*CC + k2*PCN - k4*PCC - k1*PCC - V1PC*PCC/(Kp_pcc + PCC) - kd*PCC)                   # PER/CRY CYTOSOL COMPLEX
    ode[5]  = k1*PCC - k2*PCN  - V3PC*PCN/(Kp_pcn + PCN) - kd*PCN                                        # PER/CRY NUCLEUS
    ode[6]  = ksB*MB0 - k5*BC + k6*BN - V1B*BC/(Kp_bc + BC) - kd*BC                                      # BMAL CYTOSOL
    ode[7]  = (k5*BC-k6*BN-V3B*BN/(Kac_bn+BN) + V4B*NAD*BNac/(Kdac_bn+BNac)-vdBN*BN/(Kd_bn+BN) -kd*BN)   # BMAL1 NUCLEUS
    ode[8]  = V3B*BN/(Kac_bn+BN) -V4B*NAD*BNac/(Kdac_bn+BNac)-kd*BNac                                    # BMAL1_ACETYL
    ode[9]  = (v0n+vsn*BNac**n_n/(KAN**n_n*(1+(PCN/Rn)**r_n)+BNac**n_n) - vmN*MN/(KmN+MN) - kd*MN)       # Nampt
    ode[10] = ksN*MN - vm_NAMPT*N/(Km_NAMPT+N) - kd*N                                                    # NAMPT PROTEIN
    ode[11] = sn*N - vm_NAD*NAD/(Km_NAD+NAD) - kd*NAD                                                    # NAD+ cellular levels (~deacetylase activity)

    ode     = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))
    fn.setOption("name","Pegy's Model")
    
    return fn


def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)
