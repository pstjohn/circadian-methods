
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import component as c
import recomposer as r

class Decomposition():
    """The Decomposition class is the result of a decomposition procedure.
       Instances can summarize the decomposition as well as reconstitute
       data series based on their components."""
    
    def __init__(self, specs, bias=0.0):
        """Initializes a decomposition object. Demeter: save copies of 
           component elements for wrapper-level querying."""
        self._components = []
        
        self.bias = bias
        self.specs = specs
        for spec in specs:
            # For historical reasons, given as: (amp, decay, freq, phase)
            amp, decay, freq, ph = spec
            self._components.append(c.Component(amp, decay, freq, phase=ph))
        
        self.freqs = [comp.freq for comp in self._components]
        self.amps = [comp.amp for comp in self._components]
        self.decays = [comp.decay for comp in self._components]
        self.periods = [comp.period for comp in self._components]
        self.phases = [comp.phase for comp in self._components]
        self.offsets = [comp.offset for comp in self._components]
    
    def time_series(self, length):
        """Builds a recomposed time series based on all components."""
        maker = r.Recomposer(self._components, self.bias)
        return maker.time_series(length)
    
    def period_limit_time_series(self, length, period, use_smalls=False):
        """Actually, much more useful than filtered time series option.
           Enables to set period cap/floor on recomposition components."""
        filtered = self._components[:]
        if use_smalls:
            filtered = filter(lambda c: c.period <= period, filtered)
        else:
            filtered = filter(lambda c: c.period >= period, filtered)
        
        maker = r.Recomposer(filtered, self.bias)
        return maker.time_series(length)
    
    def filtered_time_series(self, length, n, should_reverse=False):
        """Generates a time series based on a subset of components,
           specified by a slice tuple. Assumed to be period sorted..."""
        sorts = self._components[:]
        sorts.sort(key=lambda c: c.period, reverse=should_reverse)
        if isinstance(n, tuple) and len(n) == 2:
            filtered = sorts[n[0]:n[1]]
        elif isinstance(n, int):
            filtered = sorts[:n]
        else:
            raise Exception("Please argue n as integer or slice tuple!")
        maker = r.Recomposer(filtered, self.bias)
        return maker.time_series(length)
    
    def count(self):
        """Returns the number of components in the decomposition"""
        return len(self._components)
    
    def summary(self):
        """Returns a concise summary of decomposition components."""
        return [c.to_tuple() for c in self._components]

    def pseudo_r2(self):
        """Calculates the pseudo-r2 value for the fitted components."""
        y_reg = self.time_series(len(self.data))
        SSres = ((self.data - y_reg)**2).sum()
        SStot = ((self.data - self.data.mean())**2).sum()
        return 1 - SSres/SStot
        
if __name__ == "__main__":
    print "This module defines a class for component-wise Decompositions."
