
import sys
import os.path
import random
from math import cos, pi, e

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class Periodic:
    """Periodic generates timeseries values for a given periodic signal:
       \sum_{k=0}^{M-1} { a_k e^{s_k n} + w(n) }
       i.e. composed of M exponentiall-damped, complex-valued sinusoids."""
    
    def __init__(self, specs=None, **parameters):
        """Initialization requires either a number of components or a
           definite spec for individual sinusoidal components."""
        
        if "noise" in parameters.keys():
            noise = parameters["noise"]
        else:
            noise = 0.5 * random.random()
        
        if specs:
            pass
        elif "count" in parameters.keys():
            count = parameters["count"]
        else:
            raise Exception("Must specify either specs or count in Periodic!")
        
        self.specs = specs or [self._make_spec() for i in range(count)]
        self.noise = noise

    def _make_spec(self):
        """a_k e^{s_k n} + w(n) \where
           s_k = -\alpha{}_k + i2\pi{}f_k"""
        a = 20.0*random.random()
        alpha = 2 * random.random()
        period = float(random.randint(20,150))
        f = 1. / period
        offset = random.randint(0, period-1)
        return (a, alpha, f, offset)
    
    def time_series(self, length):
        """Builds a time-domain series composed of the defined sinusoids."""
        if length <= 0:
            raise Exception("time_series length must be stricly positive.")
        return self._make_time_series(length)
    
    def _make_time_series(self, length):
        times = range(length)
        series = [self._calculate_sum_at(t) for t in times]
        return series
    
    def _calculate_sum_at(self, time):
        signals = [self._calculate_spec_at(s, time) for s in self.specs]
        return sum(signals)
    
    def _calculate_spec_at(self, spec, time):
        a, alpha, f, offset = spec
        
        damp = e ** (-1 * alpha * time)
        signal = cos(2 * pi * f * (time - offset))
        noise = self.noise * random.gauss(0.0, 1.0)
        return a * damp * (signal + noise)
    
class Gaussian:
    """Generates gaussian noise time-series."""
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
    
    def time_series(self, length):
        if length <= 0:
            raise Exception("time_series length must be stricly positive.")
        series = [random.gauss(self.mean, self.stdev) for i in range(length)]
        return series

class Step:
    """Generates time-series of step functions."""
    def __init__(self, amp, per, off, bias, nsr):
        self.amplitude = amp
        self.period = per
        self.offset = off
        self.bias = bias
        self.noise = nsr
    
    def time_series(self, length):
        if length <= 0:
            raise Exception("time_series length must be stricly positive.")
        series = [self._make_value(t) for t in range(length)]
        return series
    
    def _make_value(self, time):
        ttime = time - self.offset
        n_per = int(ttime / self.period)
        if n_per % 2 == 0:
            value = self._high_value()
        else:
            value = self._low_value()
        return value
        
    def _high_value(self):
        signal = self.bias + self.amplitude
        noise = self.noise * self.amplitude * random.gauss(0, 1.0)
        return signal + noise
    
    def _low_value(self):
        signal = self.bias - self.amplitude
        noise = self.noise * self.amplitude * random.gauss(0, 1.0)
        return signal + noise

class Impulse:
    """Generates a periodic impulse time-series."""
    def __init__(self, amp, width, per, bias, nsr):
        self.amplitude = amp
        self.width = width
        self.period = per
        self.bias = bias
        self.noise = nsr
        
    def time_series(self, length):
        if length <= 0:
            raise Exception("time_series length must be stricly positive.")
        series = [self._get_value(t) for t in range(length)]
        return series
    
    def _get_value(self, time):
        partial = time % self.period
        dist = abs(partial - self.period)
        frac = max(0.0, float(self.width - dist) / float(self.width))
        signal = self.bias + frac * 2 * self.amplitude
        noise = self.noise * self.amplitude * random.gauss(0.0, 1.0)
        return signal + noise

if __name__ == "__main__":
    print "This is a module for defining test-signal generators."
