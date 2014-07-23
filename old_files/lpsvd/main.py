
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from numpy.linalg import lstsq, norm
from scipy.linalg import svd, hankel

from math import exp, sin, cos, atan2
from cmath import polar, pi

import decomposition as d

class LPSVD:
    """Encapsulates the logic needed to perform Linear-Prediction Singular
       Value Decomposition procedure on input time-series data..."""
    
    def __init__(self, data, count=None, counter=None, filterf=None):
        """Initializes with array-like time-series argument data."""
        self.data = np.array(data)
        self.count = count
        self.counter = counter
        self.filterf = filterf
        self.svd = None
    
    def decomposition(self):
        comps, bias = self.components()
        decomp = d.Decomposition(comps, bias)
        decomp.data = self.data
        return decomp
    
    def components(self):
        """Compute components. Collect into single spec tuples."""
        df_cs = self.half_components()
        ap_cs, bias = self.ap_components(df_cs)
        
        # df component: (decay coefficient, frequency)
        # ap component: (amplitude, phase)
        comps = [(ap[0],df[0],df[1],ap[1]) for (df,ap) in zip(df_cs,ap_cs)]
        return comps, bias

    def ap_components(self, hcs):
        """Returns tuple: (ap_components, bias), where ap_components
           has best-fit parameter amplitudes and phases for sinusoids."""
        v = self._lfit_apcs(hcs)
        
        bias = v[-1] # Here the bias appears following linear fitting step.
        pairs = [(v[2*i], v[2*i+1]) for i in range(len(v)/2)]
        # amps = [norm(p) for p in pairs]
        phases = [atan2(p[0],p[1]) for p in pairs]

        # HACK: These huge ones come from arithmetic errors in LLS fitting 0.0
        amps = []
        for p in pairs:
            if abs(p[0]) >= 1.0e4 and abs(p[1]) >= 1.0e4:
                amps.append(0)
            elif abs(p[0]) >= 1.0e5:
                amps.append(p[1])
            elif abs(p[1]) >= 1.0e5:
                amps.append(p[0])
            else:
                amps.append(norm(p))
        
        return zip(amps, phases), bias
    
    def _lfit_apcs(self, hcs):
        """Linear fit to sinusoids of known frequencies, where:
           
           A sin(freq t + phase) = a sin(freq t) + b cos(freq t)
           A = sqrt(a*a + b*b)
           phase = atan2(a,b)
           
           Allows us to linearize the phase and amplitude fitting."""
        b = np.array(self.data[:])
        ts = range(len(self.data))
        trows = [(t,[]) for t in ts]
        
        for t,row, in trows:
            for hc in hcs:
                d,f = hc
                row.append(exp(-1*d*t)*sin(f*2*pi*t))
                row.append(exp(-1*d*t)*cos(f*2*pi*t))
            else:
                row.append(1.0) # bias
        A = np.array([r[1] for r in trows])
        v = lstsq(A,b,rcond=0.05)[0]
        return v
    
    def half_components(self):
        """Yields decay coefficients and frequencies for components."""
        roots = self.half_component_roots()
        
        filterf = self.filterf
        if filterf == None:
            # by default, filter to outside unit circle
            filterf = lambda c: abs(c) >= 1.0
        
        froots = filter(filterf, roots)
        froots = filter(lambda c: c.imag >= 0, froots)
        decays = [np.log(polar(c)[0]) for c in froots]
        freqs = [polar(c)[1] / (2*pi) for c in froots]
        return zip(decays, freqs)
    
    def half_component_roots(self):
        """Half-components: These roots corresponds to the decay coefficients
           and frequencies observed in the component sinusoids."""
        coefficients = -1 * self.polynomial_coefficients()
        coefficients = np.insert(coefficients, 0, 1.0)
        return np.poly1d(coefficients).r
    
    def polynomial_coefficients(self):
        """Returns prediction polynomial coefficients. Memoizes decomposition
           in an instance attribute for future decompositions."""
        x = self.predictions_vector()
        if self.svd == None:
            X = self.prediction_matrix()
            self.svd = svd(X)
        U,s,Vh = self.svd
        
        n = self.get_signal_count(s)
        
        invs = 1.0/self.bias_filter(s,n)
        
        invSig = np.diag(invs[:n])
        a = np.dot(Vh.T[:,:n],
                   np.dot(invSig,
                          np.dot(U[:,:n].T, x)
                          )
                   )
        return a
    
    def bias_filter(self, data, count):
        """Subtract arithmetic mean of noise-related singular values from
           the singular value vector. Bias correction procedure."""
        if count >= len(data):
            return data
        signals = data[:count]
        remains = data[count:]
        bias = np.average(remains)
        return signals - bias
    
    def get_signal_count(self, values):
        """Prioritizes signal counting techniques: defers to an imposed
           count, if provided; delegates to a counter, if available;
           otherwise uses naive internal counting method."""
        count = len(values)
        if self.count != None:
            count = self.count
        elif self.counter != None:
            count = self.counter.count_signals(values)
        else:
            count = self._count_signals(values)
        return count
    
    def _count_signals(self, values):
        """Naive singular value selection method. Actual selection should
           be done with counter objects. See counters.py for classes."""
        nsignals = int(0.5 * len(values))
        if nsignals % 2 == 1:
            nsignals += 1
        return nsignals
    
    def predictions_vector(self):
        """Linear Predictions in the reverse direction are the initial
           values in the time-series data, which are back predicted."""
        N = len(self.data)
        M = N - int(0.75 * N)
        return np.array(self.data[:M])
    
    def prediction_matrix(self):
        """Reverse linear prediction LxM prediction matrix, where:
           L = 0.75 * N and M = 0.25 * N"""
        N = len(self.data)
        L = int(0.75 * N)
        return hankel(self.data[1:N-L+1], self.data[-L:])

if __name__ == "__main__":
    print "This is a module for Linear Prediction SVD."
