
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import random

def make_specs(ncomps):
    """Utility for randomly generating Periodic [signaller] specs."""
    amps = [random.random() * 10.0 for i in range(ncomps)]
    damps = [random.random() / 60.0 for i in range(ncomps)]
    freqs = [1.0 / random.randrange(20.0, 200.0) for i in range(ncomps)]
    offsets = [random.randint(0,50) for i in range(ncomps)]
    
    specs = zip(amps, damps, freqs, offsets)
    return specs

def mean_sq_error(lista, listb):
    """Easy calculation for mean squared error between two series."""
    pairs = zip(lista, listb)
    sq_errors = [(a - b)**2 for (a,b) in pairs]
    return sum(sq_errors) / float(len(pairs))

def deltas(target, actual):
    """Provides a list of signed deltas between target and actual series."""
    pairs = zip(target, actual)
    return [(a-t) for (t,a) in pairs]

def errors(target, actual):
    """Signed and normalized errors between target and actual series."""
    pairs = zip(target, actual)
    return [(a-t)/t for (t,a) in pairs if t != 0]

if __name__ == "__main__":
    print "This module provides generic utilities."
