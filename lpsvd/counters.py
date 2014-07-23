
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class Counter():
    """Abstract base class. Inherited classes will encapsulate logic
       for identifying signal vs. noise singular values."""
    pass

class RelativeMassCounter(Counter):
    """Accepts singular values up to a certain mass of maximal SV."""
    def __init__(self, fraction):
        """Initializes with a fractional value that determines SV cutoff."""
        self.fraction = fraction
    
    def count_signals(self, values):
        max_sv = max(values)
        cutoff = self.fraction * max_sv
        signals = filter(lambda v: v > cutoff, values)
        nsignals = len(signals)
        if nsignals % 2 == 1:
            nsignals += 1
        return min(nsignals, len(values))

class MassFractionCounter(Counter):
    """Accepts a simple mass fraction of the singular values."""
    def __init__(self, fraction):
        """Initializes with a mass fraction of singular values to accept."""
        self.fraction = fraction
    
    def count_signals(self, values):
        nsignals = 0
        v_sum = sum(values)
        s_sum = 0.0
        for v in values:
            s_sum += v
            nsignals += 1
            if s_sum / v_sum > self.fraction:
                break
        if nsignals % 2 == 1:
            nsignals += 1
        return min(nsignals, len(values))

if __name__ == "__main__":
    print "This is a module for defining Singular Value counters."
