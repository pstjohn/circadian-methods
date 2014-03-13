import numpy as np
import casadi as cs

from CommonFiles import pBase

class ResponseCurves(pBase):

    def calc_adjoint(self, res):
        ts = np.linspace(5*self.y0[-1], 6*self.y0[-1], num=res)
        integrator = cs.CVodesIntegrator(self.modlT)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", 1)
        integrator.setOption("numeric_jacobian", True)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()

        seed = [1] + [0]*(self.NEQ-1)
        y0 = self.y0[:-1]

        integrator.setInput(y0, cs.INTEGRATOR_X0)

        adjoint_sensitivities = []
        ys = []
        dfdp = []
        param = list(self.paramset) + [ts[0]]

        for param[-1] in ts:       
            integrator.setInput(param, cs.INTEGRATOR_P)
            integrator.setAdjSeed(seed, cs.INTEGRATOR_XF)
            integrator.evaluate(0, 1)
            adjsens = integrator.adjSens(cs.INTEGRATOR_X0).toArray().flatten()
            adjoint_sensitivities.append(adjsens)   
            ys.append(integrator.output().toArray().flatten())
            dfdp.append(self.dfdp(ys[-1]))


        Q = np.array(adjoint_sensitivities)
        dfdp = np.array(dfdp)
        self.VRC = np.array([Q[i].dot(dfdp[i]) for i in xrange(len(Q))])
        self.relVRC = self.VRC*np.array(self.paramset)
        

    def calc_fundamental_matrices(self, res):

        self.limitCycle()
        # Some constants
        ABSTOL = 1e-11
        RELTOL = 1e-9
        MAXNUMSTEPS = 80000
        SENSMETHOD = "staggered"
        
        dt = self.y0[-1]/res

        integrator = cs.CVodesIntegrator(self.model)
        integrator.setOption("abstol",ABSTOL)
        integrator.setOption("reltol",RELTOL)
        integrator.setOption("max_num_steps",MAXNUMSTEPS)
        integrator.setOption("sensitivity_method",SENSMETHOD);
        integrator.setOption("t0", 0)
        integrator.setOption('tf', dt)
        integrator.setOption("numeric_jacobian",True)
        integrator.setOption("fsens_err_con",1)
        integrator.setOption("fsens_abstol",RELTOL)
        integrator.setOption("fsens_reltol",ABSTOL)

        integrator.init()
        
        phi_i = []
        for i in xrange(res):
            integrator.setInput(self.lc(i*dt), cs.INTEGRATOR_X0)
            integratordyfdy0 = integrator.jacobian(cs.INTEGRATOR_X0,
                                                   cs.INTEGRATOR_XF)
            integratordyfdy0.setInput(self.lc(i*dt), cs.INTEGRATOR_X0)
            integratordyfdy0.init()
            integratordyfdy0.evaluate()
            phi_i += [integratordyfdy0.output().toArray()]

        phi_i = np.array(phi_i)
            






        # 
        # sim = cs.Simulator(integrator, ts)
        # sim.init()
        # sim.setInput(self.y0[:-1],cs.INTEGRATOR_X0)
        # sim.setInput(self.paramset,cs.INTEGRATOR_P)
        # simdyfdy0 = sim.jacobian(cs.INTEGRATOR_X0,
        #                          cs.INTEGRATOR_XF)
        # simdyfdy0.evaluate()
        # phi_i = simdyfdy0.output().toArray().reshape([len(ts), self.NEQ,
        #                                               self.NEQ])
        # y_i = sim.output().toArray()


        return phi_i

    def calc_phi_T(self, res):
        ts = np.linspace(0, self.y0[-1], num=res)
        phi_i, y_i = self.calc_fundamental_matrices(ts)

        phi_T = np.zeros(phi_i.shape)
        PRC = np.zeros((res, self.NEQ))

        def multiply_chain(start, stop):
            cumulative = phi_i[start]
            for i in xrange(start-stop):
                cumulative.dot(phi_i[start-(i+1)])
            return cumulative

        for i in xrange(res):
            # Fundamental matrix
            phi_T[i] = multiply_chain(i-1, 0).dot(multiply_chain(res-1, i))

            # Left eigenvector
            L = np.ones(self.NEQ)

            # Solve linear system
            b = -phi_T[i][0,1:]
            a = phi_T[i][1:,1:] - np.eye(self.NEQ-1)
            x = np.linalg.lstsq(a.T, b.T)[0]
            L[1:] = x

            PRC[i] = L/(L.T.dot(self.dydt(self.lc(dt*i))))





if __name__ == "__main__":
    from CommonFiles.tyson2statemodel import model, paramset
    new = ResponseCurves(model(), paramset)


    
