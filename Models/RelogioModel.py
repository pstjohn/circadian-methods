# Model from:
# A. Relogio, P. O. Westermark, T. Wallach, K. Schellenberg, A. Kramer,
# and H. Herzel, "Tuning the Mammalian Circadian Clock: Robust Synergy
# of Two Loops," PLoS Computational Biology, vol. 7, no. 12, p.
# e1002309, Dec. 2011.


import numpy as np
import casadi as cs

NEQ = 19
NP = 71

y0in = np.array(
    [2.04735794,   2.45086116,   2.89112647,   1.06819258,
     5.35751776,   2.42938302,   5.29613407,   0.92242458,
     1.65495036,   0.08011401,   0.52772583,   6.45275616,
     8.22162788,   0.05601451,   0.04554138,   0.29276828,
     0.37528026,   1.1034227 ,   0.59096222,  25.12787039])

paramset = np.array(
    [8.00000000e-02,   6.00000000e-02,   9.00000000e-02,
     1.70000000e-01,   1.20000000e-01,   1.50000000e-01,
     3.00000000e-01,   2.00000000e-01,   2.00000000e+00,
     2.00000000e-01,   1.60000000e+00,   2.30000000e-01,
     2.50000000e-01,   6.00000000e-01,   2.00000000e-01,
     2.00000000e-01,   3.10000000e-01,   3.00000000e-01,
     7.30000000e-01,   2.30000000e+00,   1.00000000e-02,
     1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
     1.00000000e+00,   2.00000000e+00,   5.00000000e-02,
     1.00000000e+00,   2.92000000e+00,   1.90000000e+00,
     1.09000000e+01,   1.00000000e+00,   3.00000000e+00,
     9.00000000e-01,   2.40000000e+00,   7.00000000e-01,
     5.20000000e+00,   2.07000000e+00,   3.30000000e+00,
     9.00000000e-01,   4.00000000e-01,   8.35000000e+00,
     1.94000000e+00,   1.20000000e+01,   1.20000000e+01,
     5.00000000e+00,   5.00000000e+00,   1.20000000e+01,
     4.00000000e-01,   2.60000000e-01,   3.70000000e-01,
     7.60000000e-01,   1.21000000e+00,   2.00000000e-01,
     1.00000000e-01,   5.00000000e-01,   1.00000000e-01,
     1.00000000e-01,   2.00000000e-02,   2.00000000e-02,
     5.00000000e+00,   7.00000000e+00,   6.00000000e+00,
     4.00000000e+00,   1.00000000e+00,   6.00000000e+00,
     2.00000000e+00,   6.00000000e+00,   3.00000000e+00,
     2.00000000e+00,   5.00000000e+00])

def model(): 

    # Time
    t = cs.ssym('t')

    # y-variables
    CLKBM1  = cs.ssym('CLKBM1')
    reverb  = cs.ssym('reverb')
    ror     = cs.ssym('ror')
    REVERBc = cs.ssym('REVERBc')
    RORc    = cs.ssym('RORc')
    REVERBn = cs.ssym('REVERBn')
    RORn    = cs.ssym('RORn')
    bm1     = cs.ssym('bm1')
    BM1c    = cs.ssym('BM1c')
    BM1n    = cs.ssym('BM1n')
    per     = cs.ssym('per')
    cry     = cs.ssym('cry')
    Cc      = cs.ssym('Cc')
    Pc      = cs.ssym('Pc')
    Pcp     = cs.ssym('Pcp')
    PcpCc   = cs.ssym('PcpCc')
    PcCc    = cs.ssym('PcCc')
    PnpCn   = cs.ssym('PnpCn')
    PnCn    = cs.ssym('PnCn')

    x = [CLKBM1, reverb, ror, REVERBc, RORc, REVERBn, RORn, bm1, BM1c,
         BM1n, per, cry, Cc, Pc, Pcp, PcpCc, PcCc,
         PnpCn, PnCn]
    
    # Parameters
    dCLKBM1   = cs.ssym('dCLKBM1')
    dPnpCn    = cs.ssym('dPnpCn')
    dPnCn     = cs.ssym('dPnCn')
    dREVERBn  = cs.ssym('dREVERBn')
    dRORn     = cs.ssym('dRORn')
    dBM1n     = cs.ssym('dBM1n')
    dper      = cs.ssym('dper')
    dcry      = cs.ssym('dcry')
    dreverb   = cs.ssym('dreverb')
    dror      = cs.ssym('dror')
    dbm1      = cs.ssym('dbm1')
    dCc       = cs.ssym('dCc')
    dPc       = cs.ssym('dPc')
    dPcp      = cs.ssym('dPcp')
    dPcpCc    = cs.ssym('dPcpCc')
    dPcCc     = cs.ssym('dPcCc')
    dREVERBc  = cs.ssym('dREVERBc')
    dRORc     = cs.ssym('dRORc')
    dBM1c     = cs.ssym('dBM1c')
    kfCLKBM1  = cs.ssym('kfCLKBM1')
    kdCLKBM1  = cs.ssym('kdCLKBM1')
    kfPcpCc   = cs.ssym('kfPcpCc')
    kdPcpCc   = cs.ssym('kdPcpCc')
    kfPcCc    = cs.ssym('kfPcCc')
    kdPcCc    = cs.ssym('kdPcCc')
    kphPc     = cs.ssym('kphPc')
    kdphPcp   = cs.ssym('kdphPcp')
    V1max     = cs.ssym('V1max')
    V2max     = cs.ssym('V2max')
    V3max     = cs.ssym('V3max')
    V4max     = cs.ssym('V4max')
    V5max     = cs.ssym('V5max')
    kt1       = cs.ssym('kt1')
    ki1       = cs.ssym('ki1')
    kt2       = cs.ssym('kt2')
    ki2       = cs.ssym('ki2')
    ki21      = cs.ssym('ki21')
    kt3       = cs.ssym('kt3')
    ki3       = cs.ssym('ki3')
    kt4       = cs.ssym('kt4')
    ki4       = cs.ssym('ki4')
    kt5       = cs.ssym('kt5')
    ki5       = cs.ssym('ki5')
    a         = cs.ssym('a')
    d         = cs.ssym('d')
    g         = cs.ssym('g')
    h         = cs.ssym('h')
    i         = cs.ssym('i')
    kp1       = cs.ssym('kp1')
    kp2       = cs.ssym('kp2')
    kp3       = cs.ssym('kp3')
    kp4       = cs.ssym('kp4')
    kp5       = cs.ssym('kp5')
    kiPcpCc   = cs.ssym('kiPcpCc')
    kiPcCc    = cs.ssym('kiPcCc')
    kiREVERBc = cs.ssym('kiREVERBc')
    kiRORc    = cs.ssym('kiRORc')
    kiBM1c    = cs.ssym('kiBM1c')
    kePnpCn   = cs.ssym('kePnpCn')
    kePnCn    = cs.ssym('kePnCn')
    b         = cs.ssym('b')
    c         = cs.ssym('c')
    e         = cs.ssym('e')
    f         = cs.ssym('f')
    f1        = cs.ssym('f1')
    v         = cs.ssym('v')
    w         = cs.ssym('w')
    p         = cs.ssym('p')
    q         = cs.ssym('q')
    n         = cs.ssym('n')
    m         = cs.ssym('m')


    param = [dCLKBM1, dPnpCn, dPnCn, dREVERBn, dRORn, dBM1n,
             dper, dcry, dreverb, dror, dbm1, dCc, dPc, dPcp,
             dPcpCc, dPcCc, dREVERBc, dRORc, dBM1c, kfCLKBM1,
             kdCLKBM1, kfPcpCc, kdPcpCc, kfPcCc, kdPcCc,
             kphPc, kdphPcp, V1max, V2max, V3max, V4max, V5max, kt1,
             ki1, kt2, ki2, ki21, kt3, ki3, kt4, ki4, kt5, ki5, a, d, g,
             h, i, kp1, kp2, kp3, kp4, kp5, kiPcpCc, kiPcCc,
             kiREVERBc, kiRORc, kiBM1c, kePnpCn, kePnCn, b, c,
             e, f, f1, v, w, p, q, n, m]
    
    # Model Equations
    ode = [[]]*NEQ

    # x = [CLKBM1, reverb, ror, REVERBc, RORc, REVERBn, RORn, bm1, BM1c,
    #      BM1n, per, cry, Cc, Pc, Pcp, PcpCc, PcCc,
    #      PnpCn, PnCn]
    
    # Algebraic Substitution
    PC = PnpCn + PnCn

    ode[0]  = kfCLKBM1*BM1n - kdCLKBM1*CLKBM1 - dCLKBM1*CLKBM1
    ode[1]  = (V3max*(1 + g*(CLKBM1/kt3)**v)/(1 +
               ((PC/ki3)**w)*((CLKBM1/kt3)**v) + (CLKBM1/kt3)**v) -
               dreverb*reverb)
    ode[2]  = (V4max*(1 + h*(CLKBM1/kt4)**p)/(1 +
               ((PC/ki4)**q)*((CLKBM1/kt4)**p) + (CLKBM1/kt4)**p) -
               dror*ror)
    ode[3]  = kp3*reverb - kiREVERBc*REVERBc - dREVERBc*REVERBc
    ode[4]  = kp4*ror - kiRORc*RORc - dRORc*RORc
    ode[5]  = kiREVERBc*REVERBc - dREVERBn*REVERBn 
    ode[6]  = kiRORc*RORc - dRORn*RORn 
    ode[7]  = V5max*(1 + i*(RORn/kt5)**n)/(1 + ((REVERBn/ki5)**m) +
                                         (RORn/kt5)**n) - dbm1*bm1
    ode[8]  = kp5*bm1 - kiBM1c*BM1c - dBM1c*BM1c
    ode[9]  = kiBM1c*BM1c + kdCLKBM1*CLKBM1 - kfCLKBM1*BM1n - dBM1n*BM1n
    ode[10] = V1max*(1 + a*(CLKBM1/kt1)**b)/(1 + ((PC/ki1)**c)*(CLKBM1/kt1)**b +
                                         (CLKBM1/kt1)**b) - dper*per
    ode[11] = V2max*((1 + d*(CLKBM1/kt2)**3)/(1 + ((PC/ki2)**f)*(CLKBM1/kt2)**e +
              (CLKBM1/kt2)**3))*(1/(1 + (REVERBn/ki21)**f1)) - dcry*cry
    ode[12] = (kp2*cry + kdPcpCc*PcpCc + kdPcCc*PcCc - kfPcCc*Cc*Pc -
               kfPcpCc*Cc*Pcp - dCc*Cc)
    ode[13] = (kp1*per + kdPcCc*PcCc + kdphPcp*Pcp - kfPcCc*Cc*Pc -
               kphPc*Pc - dPc*Pc)
    ode[14] = kphPc*Pc + kdPcpCc*PcpCc - kdphPcp*Pcp - kfPcpCc*Pcp*Cc - dPcp*Pcp
    ode[15] = kfPcpCc*Cc*Pcp + kePnpCn*PnpCn - kiPcpCc*PcpCc - kdPcpCc*PcpCc - dPcpCc*PcpCc
    ode[16] = kfPcCc*Cc*Pc + kePnCn*PnCn - kiPcCc*PcCc - kdPcCc*PcCc - dPcCc*PcCc
    ode[17] = kiPcpCc*PcpCc - kePnpCn*PnpCn - dPnpCn*PnpCn
    ode[18] = kiPcCc*PcCc - kePnCn*PnCn - dPnCn*PnCn

    ode = cs.vertcat(ode)
    x = cs.vertcat(x)
    param = cs.vertcat(param)

    fn = cs.SXFunction(cs.daeIn(t=t, x=x, p=param),
                       cs.daeOut(ode=ode)) 


    fn.setOption("name","Relogio2011")
    
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


