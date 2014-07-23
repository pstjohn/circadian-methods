import numpy as np
import casadi as cs

from Periodic_wrap import Periodic_wrap
    
class Evaluate(Periodic_wrap):
    """ Class to evaluate unknown y0 conditions """

    def __init__(self, model, paramset, y0=None):
        """ Passing a y0 value of None will initiate the calculation
        routines with default options """

        if y0 is None:
            y0 = np.ones(model.input(cs.DAE_X).size() + 1)

        Periodic_wrap.__init__(model, paramset, y0)
