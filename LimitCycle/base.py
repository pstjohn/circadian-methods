import casadi as cs

class base(object):
    """
    Base class to handle Casadi ODE functions. Provides easy access to
    state variables and parameters. More advanced classes should inheret
    from here.
    """

    def __init__(self, model, paramset, y0):
        """ Basic call signature for LimitCycle classes """
    
        self.model = model
        self.NEQ = self.model.input(cs.DAE_X).size()
        self.NP  = self.model.input(cs.DAE_P).size()

        self.model.init()
        self.jacp = self.model.jacobian(cs.DAE_P,0); self.jacp.init()
        self.jacy = self.model.jacobian(cs.DAE_X,0); self.jacy.init()

        self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.NEQ)]
        self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.NP)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.NP)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.NEQ)):
            self.ydict[par] = ind

        self.modlT = _modify_model(model)

        self.paramset = paramset

        assert len(paramset) == self.NP, \
                "paramset length (%d) is not the same as model \
                NP (%d)" % (len(paramset), self.NP)

        assert len(y0) == self.NEQ + 1, \
                "y0 length (%d) is not the same as model \
                NEQ + 1 (%d)" % (len(y0), self.NEQ)


def _modify_model(model):
    """
    Creates a new casadi model with period as a parameter, such that
    the model has an oscillatory period of 1. Necessary for the
    exact determinination of the period and initial conditions
    through the BVP method. (see Wilkins et. al. 2009 SIAM J of Sci
    Comp)
    """

    pSX = model.inputSX(cs.DAE_P)
    T = cs.ssym("T")
    pTSX = cs.vertcat([pSX, T])
    
    modlT = cs.SXFunction(
        cs.daeIn(t=model.inputSX(cs.DAE_T),
                 x=model.inputSX(cs.DAE_X), p=pTSX),
        cs.daeOut(ode=cs.vertcat([model.outputSX()*T])))

    modlT.setOption("name", "T-shifted model")
    return modlT

