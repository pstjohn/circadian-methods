import numpy as np

from CommonFiles import Identifiability, pBase
from CommonFiles.Models.tyson2statemodel import model, y0in, paramset

from CommonFiles.Futures import map_



nk = 10
maxrelerror = 0.15
maxsyserror = 0.10

np.random.seed(1)

orig = pBase(model(), paramset, y0in)
orig.limitCycle()

ts = np.linspace(0, 24., nk, endpoint=False)
sol = orig.lc(ts * y0in[-1] / 24.)

A = np.eye(orig.NEQ)

def get_random(shape):
    return (np.random.rand(*shape) - 0.5)

y_exact = np.array([A.dot(sol[i]) for i in xrange(len(sol))]) 
y_error = (y_exact * (1 + maxrelerror * get_random(y_exact.shape) +
                      maxsyserror * y_exact.max(0) *
                      get_random(y_exact.shape)))
errors = (maxrelerror*y_exact + maxsyserror*y_exact.max(0))



IdentifyClass = Identifiability(model())
IdentifyClass.CollOpt['TF'] = 24.
IdentifyClass.CollOpt['NK'] = 10
IdentifyClass.CollOpt['PreParamEst'] = True
IdentifyClass.CollOpt['PARMIN'] = np.array(paramset)/10.
IdentifyClass.CollOpt['PARMAX'] = np.array(paramset)*10.
# IdentifyClass.CollOpt['stability'] = 10.
# IdentifyClass.CollOpt['max_cpu_time'] = 0.1

A = np.eye(IdentifyClass.NEQ)
IdentifyClass.MeasurementMatrix(A)

for i in xrange(IdentifyClass.NM):
    IdentifyClass.AttachMeasurementData(i, ts, y_exact[:,i], errors[:,i])
 

nruns = 15
IdentifyClass.CreateSeedData(nruns, method='random_uniform')

def eval_fn(run_id):
    return IdentifyClass.CollocationSolve(run_id)


if __name__ == "__main__":

    results = map_(eval_fn, xrange(nruns))

    IdentifyClass.ProcessResults(results)
    import pylab as plt
    # fig, axmatrix = plt.subplots(nrows=2, ncols=1)
    # IdentifyClass.plot_par_whisker(axmatrix[0], params=range(5))
    # IdentifyClass.plot_sens_whisker(axmatrix[1], params=range(5))
    # IdentifyClass.plot_ts_lines(ax, [0,1], rasterize=False)
    # IdentifyClass.plot_data_boxes(ax, [0,1])
    # plt.show()
    IdentifyClass.IdentifiabilityPlot()
    plt.show()

    # IdentifyClass.save_data()
