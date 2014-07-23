
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from math import cos,pi,e

class Component():
    """Class for defining individual sinusoidal components, where components
       are described by: 
       { a_k e^{s_k} }, s_k \in \mathbf{C}
       i.e. a complex-valued, exponentially-damped sinusoid."""
    
    def __init__(self, amp, decay, freq, **kwargs):
        """Initialize Components with floating point data: amp,decay,freq."""
        
        is_float = lambda v: isinstance(v, float)
        if not all(map(is_float, [amp,decay,freq]+kwargs.values())):
            raise TypeError("Must initialize Component with float values")
        self.amp = amp
        self.decay = decay
        self.freq = freq
        self.period = self._make_period(freq)
        
        keys = kwargs.keys()
        if ("offset" in keys) and ("phase" in keys):
            if kwargs["phase"] != self._make_phase(kwargs["offset"]):
                raise Exception("Inconsistent arguments to initialization.")
            else:
                self.offset = kwargs["offset"]
                self.phase = kwargs["phase"]
        elif "offset" in keys:
            self.offset = kwargs["offset"]
            self.phase = self._make_phase(self.offset)
        elif "phase" in keys:
            self.phase = kwargs["phase"]
            self.offset = self._make_offset(self.phase)
        else:
            self.phase = 0.0
            self.offset = 0.0
    
    def to_tuple(self, use_phase=False):
        """Returns a component tuple. use_phase=True if phases preferred."""
        parts = [self.amp, self.decay, self.freq, self.offset]
        if use_phase:
            parts[3] = self.phase
        return tuple(parts)
    
    def value(self, time):
        a,alpha,f,offset = self.amp, self.decay, self.freq, self.offset
        damp = e ** (-1*alpha*time)
        signal = cos(2*pi*f * (time - offset))
        return a * (damp * signal)
    
    def _make_phase(self, offset):
        """Computes a phase from time-unit offset."""
        while offset > self.period/2:
            offset -= self.period
        phase = (2*pi) * (offset / self.period)
        return phase
    
    def _make_offset(self, phase):
        """Computes a time-unit offset from phase offset."""
        if phase < 0:
            phase += 2*pi
        offset = self.period * (phase / (2*pi))
        return offset
    
    def _make_period(self, freq):
        """Check for division by zero, otherwise computes period."""
        p = 0.0
        if freq != 0.0:
            p = 1.0 / freq
        return p

if __name__ == "__main__":
    print "This module defines a class for defining sinusoidal components."
