# Packages
import casadi as cs
import numpy as np

ABSTOL = 1e-11
RELTOL = 1e-9
MAXNUMSTEPS = 80000
SENSMETHOD = "staggered"

from CommonFiles.pBase import pBase

class Sensitivity(pBase):
    
    def __init__(self,model,paramset,y0bvp=None):
        
        pBase.__init__(self,model,paramset,y0bvp)
        
        self.KNEQ = self.NEQ + 1
        self.period = self.y0[-1]
           
    def FirstOrder(self):
        # Calculate 
        self.MonodromyCalc()
        self.S0Calc()
        
        self.model.init()
        self.model.setInput(self.y0[:-1],cs.DAE_X)
        self.model.setInput(self.paramset,cs.DAE_P)
        self.model.evaluate()
        ydot0 = self.model.output().toArray().squeeze()
        
        def modljac(x,y):
            A = self.model.jacobian(x,y)
            A.init()
            A.setInput(self.y0[:-1],cs.DAE_X)
            A.setInput(self.paramset,cs.DAE_P)
            A.evaluate()
            return A.output().toArray()
        
        AA = modljac(1,0)
        BB = modljac(2,0)
        
        LHS = np.zeros([self.KNEQ,self.KNEQ])
        LHS[:-1,:-1] = self.M - np.eye(len(self.M))
        LHS[-1,:-1] = AA[0]
        LHS[:-1,-1] = ydot0
        
        RHS = np.zeros([self.KNEQ,self.NP])
        RHS[:-1] = -self.XS
        RHS[-1] = BB[0]
        
        unk = np.linalg.solve(LHS,RHS)
        self.S0 = unk[:-1]
        self.dTdp = unk[-1]
        self.reldTdp = self.dTdp*self.paramset/self.period
        
    def MonodromyCalc(self):
        intmono = cs.CVodesIntegrator(self.model)
    
        intmono.setOption("abstol",ABSTOL)
        intmono.setOption("reltol",RELTOL)
        intmono.setOption("max_num_steps",MAXNUMSTEPS)
        intmono.setOption("sensitivity_method",SENSMETHOD);
        intmono.setOption("t0",0)
        intmono.setOption("tf",self.period)
        intmono.setOption("numeric_jacobian",True)
        
        intmono.setOption("number_of_fwd_dir",self.NEQ)
        
        intmono.setOption("fsens_err_con",1)
        intmono.setOption("fsens_abstol",RELTOL)
        intmono.setOption("fsens_reltol",ABSTOL)
            
        intmono.init()
        intmono.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        intmono.setInput(self.paramset,cs.INTEGRATOR_P)
      
        x0_seed = np.eye(self.NEQ)
        for i in range(0,self.NEQ):
            intmono.setFwdSeed(x0_seed[i],cs.INTEGRATOR_X0,i)
            
        intmono.evaluate(self.NEQ)
        
        Monodromy = np.zeros((self.NEQ,self.NEQ))
        for i in range(0,self.NEQ):
            tempM = intmono.fwdSens(cs.INTEGRATOR_XF,i).toArray().squeeze()
            Monodromy[:,i] = tempM
        
        self.M = Monodromy
        
    def S0Calc(self):
        ints0 = cs.CVodesIntegrator(self.model)
    
        ints0.setOption("abstol",ABSTOL)
        ints0.setOption("reltol",RELTOL)
        ints0.setOption("max_num_steps",MAXNUMSTEPS)
        ints0.setOption("sensitivity_method",SENSMETHOD);
        ints0.setOption("tf",self.period)
        ints0.setOption("numeric_jacobian",True)
        
        ints0.setOption("number_of_fwd_dir",self.NP)
        
        ints0.setOption("fsens_err_con",1)
        ints0.setOption("fsens_abstol",RELTOL)
        ints0.setOption("fsens_reltol",ABSTOL)
            
        ints0.init()
        ints0.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        ints0.setInput(self.paramset,cs.INTEGRATOR_P)
      
        p0_seed = np.eye(self.NP)
        for i in range(0,self.NP):
            ints0.setFwdSeed(p0_seed[i],cs.INTEGRATOR_P,i)
            
        ints0.evaluate(self.NP)
        
        S0 = np.zeros((self.NEQ,self.NP))
        for i in range(0,self.NP):
            tempS0 = ints0.fwdSens(cs.INTEGRATOR_XF,i).toArray().squeeze()
            S0[:,i] = tempS0
        
        self.XS = S0
    
    def SecondOrder(self,t):
        integrator = cs.CVodesIntegrator(self.model)
        integrator.setOption("abstol",ABSTOL)
        integrator.setOption("reltol",RELTOL)
        integrator.setOption("max_num_steps",MAXNUMSTEPS)
        integrator.setOption("sensitivity_method",SENSMETHOD);
        integrator.setOption("tf",t)
        integrator.setOption("numeric_jacobian",False)
        integrator.init()
        integrator.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        integrator.setInput(self.paramset,cs.INTEGRATOR_P)
        
        int2 = integrator.jacobian(cs.INTEGRATOR_P,cs.INTEGRATOR_XF)
        int2.setOption("numeric_jacobian",True)    
        int2.init()
        int2.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        int2.setInput(self.paramset,cs.INTEGRATOR_P)
        
        int3 = int2.jacobian(cs.INTEGRATOR_P,cs.INTEGRATOR_XF)
        int3.init()
        int3.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        int3.setInput(self.paramset,cs.INTEGRATOR_P)
        int3.evaluate()
        
        tempout = int3.output().toArray().squeeze()
        out = np.empty((self.NEQ,self.NP,self.NP))

        for i in range(0,self.NEQ):
            start = i*self.NP
            end = (i+1)*self.NP
            out[i] = tempout[start:end]
            
        return out,int2.fwdSens().toArray(),int2.output().toArray()
            
        
    def calcExtrema(self):
        self.base.roots()
        self.Ymax = self.base.Ymax
        self.Ymin = self.base.Ymin
        self.Tmax = self.base.Tmax
        self.Tmin = self.base.Tmin
        
    def calcdSdp(self):
        self.calcExtrema()
        T = np.array([self.Tmax,self.Tmin]).flatten()
        inds = T.argsort()        
        
        
        ints = cs.CVodesIntegrator(self.model)

        ints.setOption("abstol",ABSTOL)
        ints.setOption("reltol",RELTOL)
        ints.setOption("max_num_steps",MAXNUMSTEPS)
        ints.setOption("sensitivity_method",SENSMETHOD);
        ints.setOption("tf",self.period)
        ints.setOption("number_of_fwd_dir",self.NP)
        ints.setOption("fsens_err_con",1)
        ints.setOption("fsens_abstol",RELTOL)
        ints.setOption("fsens_reltol",ABSTOL)
        ints.setOption("numeric_jacobian",True)
        
        ints.init()
        ints.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        ints.setInput(self.paramset,cs.INTEGRATOR_P)
        
        p0_seed = np.eye(self.NP)
        for i in range(0,self.NP):
            ints.setFwdSeed(p0_seed[i],cs.INTEGRATOR_P,i)
        
        for i in range(0,self.NP):
            ints.setFwdSeed(self.S0[:,i],cs.INTEGRATOR_X0,i)
        
        def outsens(t):
            ints.setFinalTime(t)
            ints.evaluate(self.NP)    
            Sout = np.zeros((self.NEQ,self.NP))
            for i in range(0,self.NP):
                tempSout = ints.fwdSens(np.INTEGRATOR_XF,i).toArray().squeeze()
                Sout[:,i] = tempSout
            print "%0.1f" % (100.*t/self.period),
            print "%"
            return Sout

        S = np.array([outsens(t) for t in T[inds]])
        Splus = np.zeros([self.NEQ,self.NP])
        Sminus = np.zeros([self.NEQ,self.NP])
        
        S = S[inds.argsort()]

        for i in range(0,self.NEQ):
            Splus[i] = S[i,i,:]
            Sminus[i] = S[i+self.NEQ,i,:]
            
        self.Samp = Splus - Sminus
        self.relSamp = (self.Samp.T / (self.Ymax - self.Ymin)).T * self.paramset
        
    def modifiedModel(self):
        import casadi as cs
        pSX = self.model.inputSX(cs.DAE_P)
        pTSX = (self.NP+1)*[[]]
        T = cs.SX("T")
        for i in range(0,self.NP):
            pTSX[i] = pSX[i]
        pTSX[self.NP] = T
        ffcn_in = cs.DAE_NUM_IN * [[]]
        ffcn_in[cs.DAE_T] = self.model.inputSX(cs.DAE_T)
        ffcn_in[cs.DAE_X] = self.model.inputSX(cs.DAE_X)
        ffcn_in[cs.DAE_P] = pTSX
        self.modlT = cs.SXFunction(ffcn_in,[self.model.outputSX()*T])
        self.modlT.setOption("name","T-shifted model")
            
    def barP(self,senstype):
        import pylab as p
        import numpy as np
        
        fig = p.figure()
        ax = fig.add_subplot(1,1,1)
        inds = np.abs(senstype).argsort()
        inds = inds.tolist()
        inds.reverse()
        width = 0.8
        ax.bar(range(0,self.NP),senstype[inds],width)

        ax.set_xlabel('Parameter')
        p.xticks(np.arange(0,self.NP) + width/2,np.array(self.plabels)[inds],rotation=90)
        return fig
        
        
