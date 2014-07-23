
import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from scipy.stats import norm

class JTK:
    """Lightweight implementation of JTK statistical test. Runs only
       a single statistical test at given period / phase."""
    
    def __init__(self, series):
        """Initialization with seed data series."""
        if isinstance(series, np.ndarray):
            self.series = series
        else:
            self.series = np.array(series, dtype='float')
        
        self.N = N = len(series)
        
        self.stdev = self._compute_stdev(N)
        self.max_score = self._compute_max_score(N)
        self.expected = self.max_score / 2.0
    
    def run_series(self, period, offset):
        """Checks time series for period/offset combination sinusoid."""
        # No Bonferroni correction for a single expected period.
        ser = self.series
        ref = self.ref_series(self.N, period, offset)
        
        s_score = self.s_score(ser, ref)
        p_score = self.p_value(s_score)
        
        return (s_score, p_score)
    
    #
    # Normal Distribution Approximation
    #
    
    def p_value(self, S):
        """Generates a p-value based on tau score S."""
        if not S:
            return 1.0
        
        M = self.max_score
        score = (np.absolute(S) + M) / 2.0
        
        a = -1.0 * (score - 0.5)
        b = -1.0 * self.expected
        c = self.stdev
        
        p = 2.0 * norm.cdf(a,loc=b,scale=c)
        return p
    
    def _compute_max_score(self, N):
        N = float(N)
        
        # specific case for single replicates
        max_score = (N**2 - N) / 2.0
        
        return max_score
    
    def _compute_stdev(self, N):
        nn = float(N)
        ns = np.ones(N, dtype='float') # behaviour data is singletons
        
        var = (nn**2 * (2*nn + 3) - np.sum(ns**2 * (2*ns + 3))) / 72.0
        sdv = np.sqrt(var)
        
        return sdv
    
    #
    # Generating a Reference Series
    #
    
    def ref_series(self, N, period, offset):
        """Builds a time-series for a given period and offset."""
        func = np.cos
        pihat = np.round(np.pi,4)
        time_to_angle = 2 * pihat / period
        
        thetas = np.arange(N, dtype='float') * time_to_angle
        dtheta = offset * time_to_angle # convention omits factor of 2.0
        
        vals = func(thetas + dtheta)
        ranks = self.ranks(vals)
        
        series = self._expand(ranks)
        return series
    
    def ranks(self, series):
        """Numpy array rank-ordering."""
        temp = series.argsort()
        ranks = np.empty(len(series),dtype='float')
        
        ranks[temp] = np.arange(len(series),dtype='float')
        return ranks / float((len(ranks) - 1))
    
    def _expand(self, values):
        """Replication of time-series based on timereps array."""
        #
        # For behaviour time-series, no replication. Return values.
        #
        return values
    
    #
    # Computing Scores
    #
    
    def s_score(self, data, ref):
        """Determines concordant/discordant pairwise relationships."""
        q = self.rel_vector(data)
        r = self.rel_vector(ref)
        s = np.sum(q * r)
        return s
    
    def rel_vector(self, series):
        """Internal comparison vector that gives pairwise relationships."""
        if isinstance(series, np.ndarray):
            z = series
        else:
            z = np.array(series, dtype='float')
        
        n = len(series)
        idxs = self._tril_indices(n)
        signs = np.sign(np.subtract.outer(z,z))
        
        return signs[idxs]
    
    def _tril_indices(self, n):
        """Indices to retrieve comparison values."""
        xs,ys = np.tril_indices(n, k=-1)
        #
        # N.B. Omitted column-wise reindexing. I think this is ok.
        #
        return (xs,ys)



if __name__ == "__main__":
    print "This is a lightweight JTK test module for use with LPSVD."
