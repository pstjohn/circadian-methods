import casadi as cs
import numpy as np


# Define Constants

# Sensitivity Constants
ABSTOL = 1e-11
RELTOL = 1e-9
MAXNUMSTEPS = 40000
SENSMETHOD = "staggered"

# Model Constants
NEQ = 21
NP = 132

y0in = np.array([  0.24331263,   0.36209381,   0.04547321,   1.65927325,
                 0.04818164,   1.12538205,   0.71083973,   0.16910662,
                 0.25112744,   5.5923077 ,   0.22643791,   1.27781025,
                 0.26679664,   2.31027761,   1.28150823,   0.06503979,
                 0.15987961,   3.87889264,   0.49213361,   2.66470197,
                 6.12308458,  23.69403417])

paramset = [0.1716607, 3.835759558, 0.0858303, 3.29492357, 0.2574911,
            2.43570418, 2.893432389, 1.2874557, 2.718127512,
            4.282113672, 1.029359906, 3.982996413, 3.360443408,
            1.98389132, 4.12224894, 0.686643, 3.547732699, 0.460391626,
            3.28095286, 3.925954364, 2.613256494, 1.658948402,
            4.845148697, 4.496162678, 4.446903335, 3.695356005,
            4.353626317, 3.668171857, 4.914882658, 3.010199754,
            4.647195693, 3.662450425, 2.590563779, 0.291371303,
            4.388326916, 4.43466159, 4.684098611, 4.710633625,
            2.227408797, 0.479615871, 4.404167565, 0.300090388,
            0.665522854, 1.92943406, 3.584877702, 3.503847604,
            1.955871422, 4.12855588, 0.015507206, 1.568397818,
            0.55710772, 2.252066926, 4.73385771, 3.428521726,
            0.839188348, 1.983478968, 1.074371046, 3.956915968,
            1.684280364, 3.111315111, 1.903251901, 4.510085024,
            2.979677955, 2.241861961, 3.314166392, 1.455844027,
            3.763289537, 0.034474123, 0.767807662, 3.594828078,
            3.443766131, 0.688939888, 2.963831211, 4.631891838,
            2.946837404, 3.571882542, 2.747271096, 3.152158144,
            3.559099508, 3.617798678, 4.713992194, 1.226729029,
            1.592670702, 0.830571444, 2.591483455, 2.473560604,
            4.29832289, 4.885401442, 3.493483236, 2.338525788,
            2.706738561, 2.085594096, 2.179580493, 0.19662453,
            0.219387527, 0.410495833, 0.599176267, 3.19237474,
            1.419701257, 1.504270498, 3.049394589, 2.38063512,
            3.942025511, 1.687577403, 1.598390175, 3.037349595,
            4.003511077, 1.388367665, 2.580501607, 0.163209807,
            3.027661773, 1.722996683, 0.308177019, 1.515391846,
            2.275283017, 3.326320306, 3.572839417, 3.121637012,
            3.814321787, 0.950945318, 1.979874478, 1.321159514,
            1.848791674, 1.365955406, 2.419972481, 0.968237277,
            2.243076048, 1.750082962, 4.325062952, 2.816593036,
            3.96918701, 3.357124251]


def model():
    # Variable Assignments
    MP1    = cs.SX("MP1")
    MP2    = cs.SX("MP2")
    MC1    = cs.SX("MC1")
    MC2    = cs.SX("MC2")
    MREV   = cs.SX("MREV")
    MCLK   = cs.SX("MCLK")
    MBM1   = cs.SX("MBM1")
    MROR   = cs.SX("MROR")
    P1     = cs.SX("P1")
    P2     = cs.SX("P2")
    C1     = cs.SX("C1")
    C2     = cs.SX("C2")
    REV    = cs.SX("REV")
    CLK    = cs.SX("CLK")
    BM1    = cs.SX("BM1")
    ROR    = cs.SX("ROR")
    P1C1   = cs.SX("P1C1")
    P2C1   = cs.SX("P2C1")
    P1C2   = cs.SX("P1C2")
    P2C2   = cs.SX("P2C2")
    CLKBM1 = cs.SX("CLKBM1")
    
    y = [MP1, MP2, MC1, MC2, MREV, MCLK, MBM1, MROR,
         P1, P2, C1, C2, REV, CLK, BM1, ROR, P1C1,
         P2C1, P1C2, P2C2, CLKBM1]

    # Parameter Assignments
    vs0P1    = cs.SX("vs0P1")
    vs1P1    = cs.SX("vs1P1")
    vs0P2    = cs.SX("vs0P2")
    vs1P2    = cs.SX("vs1P2")
    vs0C1    = cs.SX("vs0C1")
    vs1C1    = cs.SX("vs1C1")
    vs2C1    = cs.SX("vs2C1")
    vs0C2    = cs.SX("vs0C2")
    vs1C2    = cs.SX("vs1C2")
    vs2C2    = cs.SX("vs2C2")
    vs1REV   = cs.SX("vs1REV")
    vs0CLK   = cs.SX("vs0CLK")
    vs1CLK   = cs.SX("vs1CLK")
    vs0BM1   = cs.SX("vs0BM1")
    vs1BM1   = cs.SX("vs1BM1")
    vs0ROR   = cs.SX("vs0ROR")
    vs1ROR   = cs.SX("vs1ROR")
    vs2ROR   = cs.SX("vs2ROR")

    na1_P1   = cs.SX("na1_P1")
    ni1_P1   = cs.SX("ni1_P1")
    ni2_P1   = cs.SX("ni2_P1")
    ni3_P1   = cs.SX("ni3_P1")
    ni4_P1   = cs.SX("ni4_P1")
    na1_P2   = cs.SX("na1_P2")
    ni1_P2   = cs.SX("ni1_P2")
    ni2_P2   = cs.SX("ni2_P2")
    ni3_P2   = cs.SX("ni3_P2")
    ni4_P2   = cs.SX("ni4_P2")
    na1_C1   = cs.SX("na1_C1")
    na2_C1   = cs.SX("na2_C1")
    ni1_C1   = cs.SX("ni1_C1")
    ni2_C1   = cs.SX("ni2_C1")
    ni3_C1   = cs.SX("ni3_C1")
    ni4_C1   = cs.SX("ni4_C1")
    na1_C2   = cs.SX("na1_C2")
    na2_C2   = cs.SX("na2_C2")
    ni1_C2   = cs.SX("ni1_C2")
    ni2_C2   = cs.SX("ni2_C2")
    ni3_C2   = cs.SX("ni3_C2")
    ni4_C2   = cs.SX("ni4_C2")
    na1_REV  = cs.SX("na1_REV")
    ni1_REV  = cs.SX("ni1_REV")
    ni2_REV  = cs.SX("ni2_REV")
    ni3_REV  = cs.SX("ni3_REV")
    ni4_REV  = cs.SX("ni4_REV")
    na1_CLK  = cs.SX("na1_CLK")
    ni1_CLK  = cs.SX("ni1_CLK")
    na1_BM1  = cs.SX("na1_BM1")
    ni1_BM1  = cs.SX("ni1_BM1")
    na1_ROR  = cs.SX("na1_ROR")
    na2_ROR  = cs.SX("na2_ROR")
    ni1_ROR  = cs.SX("ni1_ROR")
    ni2_ROR  = cs.SX("ni2_ROR")
    ni3_ROR  = cs.SX("ni3_ROR")
    ni4_ROR  = cs.SX("ni4_ROR")

    KA1P1    = cs.SX("KA1P1")
    KI1P1    = cs.SX("KI1P1")
    KI2P1    = cs.SX("KI2P1")
    KI3P1    = cs.SX("KI3P1")
    KI4P1    = cs.SX("KI4P1")
    KA1P2    = cs.SX("KA1P2")
    KI1P2    = cs.SX("KI1P2")
    KI2P2    = cs.SX("KI2P2")
    KI3P2    = cs.SX("KI3P2")
    KI4P2    = cs.SX("KI4P2")
    KA1C1    = cs.SX("KA1C1")
    KA2C1    = cs.SX("KA2C1")
    KI1C1    = cs.SX("KI1C1")
    KI2C1    = cs.SX("KI2C1")
    KI3C1    = cs.SX("KI3C1")
    KI4C1    = cs.SX("KI4C1")
    KA1C2    = cs.SX("KA1C2")
    KA2C2    = cs.SX("KA2C2")
    KI1C2    = cs.SX("KI1C2")
    KI2C2    = cs.SX("KI2C2")
    KI3C2    = cs.SX("KI3C2")
    KI4C2    = cs.SX("KI4C2")
    KA1REV   = cs.SX("KA1REV")
    KI1REV   = cs.SX("KI1REV")
    KI2REV   = cs.SX("KI2REV")
    KI3REV   = cs.SX("KI3REV")
    KI4REV   = cs.SX("KI4REV")
    KA1CLK   = cs.SX("KA1CLK")
    KI1CLK   = cs.SX("KI1CLK")
    KA1BM1   = cs.SX("KA1BM1")
    KI1BM1   = cs.SX("KI1BM1")
    KA1ROR   = cs.SX("KA1ROR")
    KA2ROR   = cs.SX("KA2ROR")
    KI1ROR   = cs.SX("KI1ROR")
    KI2ROR   = cs.SX("KI2ROR")
    KI3ROR   = cs.SX("KI3ROR")
    KI4ROR   = cs.SX("KI4ROR")

    kdmP1    = cs.SX("kdmP1")
    kdmP2    = cs.SX("kdmP2")
    kdmC1    = cs.SX("kdmC1")
    kdmC2    = cs.SX("kdmC2")
    kdmREV   = cs.SX("kdmREV")
    kdmCLK   = cs.SX("kdmCLK")
    kdmBM1   = cs.SX("kdmBM1")
    kdmROR   = cs.SX("kdmROR")

    tlP1     = cs.SX("tlP1")
    tlP2     = cs.SX("tlP2")
    tlC1     = cs.SX("tlC1")
    tlC2     = cs.SX("tlC2")
    tlREV    = cs.SX("tlREV")
    tlCLK    = cs.SX("tlCLK")
    tlBM1    = cs.SX("tlBM1")
    tlROR    = cs.SX("tlROR")

    upP1     = cs.SX("upP1")
    upP2     = cs.SX("upP2")
    upC1     = cs.SX("upC1")
    upC2     = cs.SX("upC2")
    upREV    = cs.SX("upREV")
    upCLK    = cs.SX("upCLK")
    upBM1    = cs.SX("upBM1")
    upROR    = cs.SX("upROR")

    arP1C1   = cs.SX("arP1C1")
    arP1C2   = cs.SX("arP1C2")
    arP2C1   = cs.SX("arP2C1")
    arP2C2   = cs.SX("arP2C2")
    arCLKBM1 = cs.SX("arCLKBM1")

    drP1C1   = cs.SX("drP1C1")
    drP1C2   = cs.SX("drP1C2")
    drP2C1   = cs.SX("drP2C1")
    drP2C2   = cs.SX("drP2C2")
    drCLKBM1 = cs.SX("drCLKBM1")

    ni5_C1   = cs.SX("ni5_C1")
    ni5_C2   = cs.SX("ni5_C2")
    ni5_ROR  = cs.SX("ni5_ROR")
    KI5C1    = cs.SX("KI5C1")
    KI5C2    = cs.SX("KI5C2")
    KI5ROR   = cs.SX("KI5ROR")

    param = [vs0P1, vs1P1, vs0P2, vs1P2, vs0C1, vs1C1, vs2C1, vs0C2,
             vs1C2, vs2C2, vs1REV, vs0CLK, vs1CLK, vs0BM1, vs1BM1,
             vs0ROR, vs1ROR, vs2ROR, na1_P1, ni1_P1, ni2_P1, ni3_P1,
             ni4_P1, na1_P2, ni1_P2, ni2_P2, ni3_P2, ni4_P2, na1_C1,
             na2_C1, ni1_C1, ni2_C1, ni3_C1, ni4_C1, na1_C2, na2_C2,
             ni1_C2, ni2_C2, ni3_C2, ni4_C2, na1_REV, ni1_REV, ni2_REV,
             ni3_REV, ni4_REV, na1_CLK, ni1_CLK, na1_BM1, ni1_BM1,
             na1_ROR, na2_ROR, ni1_ROR, ni2_ROR, ni3_ROR, ni4_ROR,
             KA1P1, KI1P1, KI2P1, KI3P1, KI4P1, KA1P2, KI1P2, KI2P2,
             KI3P2, KI4P2, KA1C1, KA2C1, KI1C1, KI2C1, KI3C1, KI4C1,
             KA1C2, KA2C2, KI1C2, KI2C2, KI3C2, KI4C2, KA1REV, KI1REV,
             KI2REV, KI3REV, KI4REV, KA1CLK, KI1CLK, KA1BM1, KI1BM1,
             KA1ROR, KA2ROR, KI1ROR, KI2ROR, KI3ROR, KI4ROR, kdmP1,
             kdmP2, kdmC1, kdmC2, kdmREV, kdmCLK, kdmBM1, kdmROR, tlP1,
             tlP2, tlC1, tlC2, tlREV, tlCLK, tlBM1, tlROR, upP1, upP2,
             upC1, upC2, upREV, upCLK, upBM1, upROR, arP1C1, arP1C2,
             arP2C1, arP2C2, arCLKBM1, drP1C1, drP1C2, drP2C1, drP2C2,
             drCLKBM1, ni5_C1, ni5_C2, ni5_ROR, KI5C1, KI5C2, KI5ROR]
         
    # Time Variable
    t = cs.SX("t")
    
    
    ode = [[]]*NEQ
    
    ode[0]= (vs0P1+vs1P1*(CLKBM1**na1_P1)/(KA1P1**na1_P1+CLKBM1**na1_P1))*(KI1P1**ni1_P1)/(KI1P1**ni1_P1+P1C1**ni1_P1)*(KI2P1**ni2_P1)/(KI2P1**ni2_P1+P1C2**ni2_P1)*(KI3P1**ni3_P1)/(KI3P1**ni3_P1+P2C1**ni3_P1)*(KI4P1**ni4_P1)/(KI4P1**ni4_P1+P2C2**ni4_P1)-kdmP1*MP1
    ode[1]= (vs0P2+vs1P2*(CLKBM1**na1_P2)/(KA1P2**na1_P2+CLKBM1**na1_P2))*(KI1P2**ni1_P2)/(KI1P2**ni1_P2+P1C1**ni1_P2)*(KI2P2**ni2_P2)/(KI2P2**ni2_P2+P1C2**ni2_P2)*(KI3P2**ni3_P2)/(KI3P2**ni3_P2+P2C1**ni3_P2)*(KI4P2**ni4_P2)/(KI4P2**ni4_P2+P2C2**ni4_P2)-kdmP2*MP2
    ode[2]= (vs0C1+vs1C1*(CLKBM1**na1_C1)/(KA1C1**na1_C1+CLKBM1**na1_C1)+vs2C1*(ROR**na2_C1)/(KA2C1**na2_C1+ROR**na2_C1))*(KI1C1**ni1_C1)/(KI1C1**ni1_C1+P1C1**ni1_C1)*(KI2C1**ni2_C1)/(KI2C1**ni2_C1+P1C2**ni2_C1)*(KI3C1**ni3_C1)/(KI3C1**ni3_C1+P2C1**ni3_C1)*(KI4C1**ni4_C1)/(KI4C1**ni4_C1+P2C2**ni4_C1)*(KI5C1**ni5_C1)/(KI5C1**ni5_C1+REV**ni5_C1)-kdmC1*MC1
    ode[3]= (vs0C2+vs1C2*(CLKBM1**na1_C2)/(KA1C2**na1_C2+CLKBM1**na1_C2)+vs2C2*(ROR**na2_C2)/(KA2C2**na2_C2+ROR**na2_C2))*(KI1C2**ni1_C2)/(KI1C2**ni1_C2+P1C1**ni1_C2)*(KI2C2**ni2_C2)/(KI2C2**ni2_C2+P1C2**ni2_C2)*(KI3C2**ni3_C2)/(KI3C2**ni3_C2+P2C1**ni3_C2)*(KI4C2**ni4_C2)/(KI4C2**ni4_C2+P2C2**ni4_C2)*(KI5C2**ni5_C2)/(KI5C2**ni5_C2+REV**ni5_C2)-kdmC2*MC2
    ode[4]= vs1REV*(CLKBM1**na1_REV)/(KA1REV**na1_REV+CLKBM1**na1_REV)*(KI1REV**ni1_REV)/(KI1REV**ni1_REV+P1C1**ni1_REV)*(KI2REV**ni2_REV)/(KI2REV**ni2_REV+P1C2**ni2_REV)*(KI3REV**ni3_REV)/(KI3REV**ni3_REV+P2C1**ni3_REV)*(KI4REV**ni4_REV)/(KI4REV**ni4_REV+P2C2**ni4_REV)-kdmREV*MREV
    ode[5]= (vs0CLK+vs1CLK*(ROR**na1_CLK)/(KA1CLK**na1_CLK+ROR**na1_CLK))*(KI1CLK**ni1_CLK)/(KI1CLK**ni1_CLK+REV**ni1_CLK)-kdmCLK*MCLK
    ode[6]= (vs0BM1+vs1BM1*(ROR**na1_BM1)/(KA1BM1**na1_BM1+ROR**na1_BM1))*(KI1BM1**ni1_BM1)/(KI1BM1**ni1_BM1+REV**ni1_BM1)-kdmBM1*MBM1
    ode[7]= (vs0ROR+vs1ROR*(CLKBM1**na1_ROR)/(KA1ROR**na1_ROR+CLKBM1**na1_ROR)+vs2ROR*(ROR**na2_ROR)/(KA2ROR**na2_ROR+ROR**na2_ROR))*(KI1ROR**ni1_ROR)/(KI1ROR**ni1_ROR+P1C1**ni1_ROR)*(KI2ROR**ni2_ROR)/(KI2ROR**ni2_ROR+P1C2**ni2_ROR)*(KI3ROR**ni3_ROR)/(KI3ROR**ni3_ROR+P2C1**ni3_ROR)*(KI4ROR**ni4_ROR)/(KI4ROR**ni4_ROR+P2C2**ni4_ROR)*(KI5ROR**ni5_ROR)/(KI5ROR**ni5_ROR+REV**ni5_ROR)-kdmROR*MROR
    ode[8]= tlP1*MP1 - upP1*P1 - arP1C1*P1*C1 - arP1C2*P1*C2 + drP1C1*P1C1 + drP1C2*P1C2
    ode[9]= tlP2*MP2 - upP2*P2 - arP2C1*P2*C1 - arP2C2*P2*C2 + drP2C1*P2C1 + drP2C2*P2C2
    ode[10]= tlC1*MC1 - upC1*C1 - arP1C1*P1*C1 - arP2C1*P2*C1 + drP1C1*P1C1 + drP2C1*P2C1
    ode[11]= tlC2*MC2 - upC2*C2 - arP1C2*P1*C2 - arP2C2*P2*C2 + drP1C2*P1C2 + drP2C2*P2C2
    ode[12]= tlREV*MREV - upREV*REV
    ode[13]= tlCLK*MCLK - upCLK*CLK - arCLKBM1*CLK*BM1 + drCLKBM1*CLKBM1
    ode[14]= tlBM1*MBM1 - upBM1*BM1 - arCLKBM1*CLK*BM1 + drCLKBM1*CLKBM1
    ode[15]= tlROR*MROR - upROR*ROR
    ode[16]= arP1C1*P1*C1 - drP1C1*P1C1
    ode[17]= arP2C1*P2*C1 - drP2C1*P2C1
    ode[18]= arP1C2*P1*C2 - drP1C2*P1C2
    ode[19]= arP2C2*P2*C2 - drP2C2*P2C2
    ode[20]= arCLKBM1*CLK*BM1 - drCLKBM1*CLKBM1

    param = cs.vertcat(param)
    y = cs.vertcat(y)
    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=param),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Henry's Model")
    
    return fn


def create_class():
    from .. import pBase
    return pBase(model(), paramset, y0in)
