
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import component as c
from math import cos,pi,e

class Recomposer:
    """Generates timeseries values given for a sinusoidal signal:
       \sum_{k=0}^{M-1} { a_k e^{s_k n} }
       i.e composed of M exponentiall-damped, complex-valued sinusoids."""
    
    def __init__(self, specs, bias=0.0):
        """Initialize with components to be evaluated and optional bias."""
        is_component = lambda comp: isinstance(comp, c.Component)
        if not all(map(is_component, specs)):
            raise TypeError("Must initialize Recomposer with Components.")
        self.specs = specs
        self.bias = bias
    
    def time_series(self, length):
        """Generates a time series of given length from the components."""
        if length <= 0:
            raise Exception("Time series lengths must be positive.")
        return self._make_time_series(length)
    
    def _make_time_series(self, length):
        times = range(length)
        unbiased = [self._calculate_sum_at(t) for t in times]
        series = [v+self.bias for v in unbiased]
        return series
    
    def _calculate_sum_at(self, time):
        signals = [self._calculate_spec_at(s, time) for s in self.specs]
        return sum(signals)
    
    def _calculate_spec_at(self, spec, time):
        # a,alpha,f,offset = spec.to_tuple()
        #
        # damp = e ** (-1*alpha*time)
        # signal = cos(2*pi*f * (time - offset))
        # return a * (damp * signal)
        return spec.value(time)

if __name__ == '__main__':
    print "This is a module for regenerating sinusoidal time series."
