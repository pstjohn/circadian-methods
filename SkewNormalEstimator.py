import numpy as np
from scipy.stats import multivariate_normal
from scipy import stats
from scipy.special import erf, cbrt
import pymc as pm 
import cPickle as pickle

import matplotlib.pyplot as plt

old_settings = np.seterr(all='ignore')

class SkewEstimator(object):
    """ Class to use bayesian regression to estimate the mean,
    covariance, and skew for a given dataset """

    def __init__(self, data, labels=None, name='SkewEstimator'):
        """ Initialize the class with data in vals. """

        self.data = data
        self.name = name

        self.n, self.p = data.shape

        # Add labels to describe each data column
        if not labels:
            self.labels = ['comp' + str(i) for i in xrange(self.p)]
        else:
            assert len(labels) == self.p, "Not enough labels"
            self.labels = labels

        # Use frequentist methods to get estimates for the mean,
        # covariance, and skew:
        self.est = {
            'mean' : self.data.mean(0),
            'cov'  : np.cov(self.data.T),
            'skew' : np.array([min([s, 0.95]) for s in
                               stats.skew(self.data)]),
        }

        # Make sure the current estimated point is valid in the DP space
        try: 
            direct_from_centered(
                self.est['mean'], self.est['cov'], self.est['skew'])

        except (AssertionError, np.linalg.linalg.LinAlgError): 
            self.est['skew'] = np.zeros(self.p)

    def _create_pymc_model(self):
        """ Initialize pymc objects using estimated values """

        # Here we give reasonably vague priors, initializing at the
        # point we estimated using frequentist methods
        cov = pm.Wishart('cov', n=(self.p*(self.p + 1))/2,
                         Tau=np.eye(self.p), value=self.est['cov'])
        mu = pm.Normal('mu', mu=0, tau=1e-5, size=self.p,
                       value=self.est['mean'])
        skew = pm.Uniform('skew', lower=-1, upper=1, size=self.p,
                          value=self.est['skew'])

        @pm.stochastic(observed=True)
        def data(value=self.data, mu=mu, cov=cov, skew=skew):
            """ Using centered parameters, calculate the log-likelyhood.
            In the event that the centered parameters do not map to
            direct parameters, return an infeasible solution """

            try:
                log_likelyhood = logmvsn(value, mean=mu, cov=cov,
                                         skew=skew)
                return log_likelyhood.sum()
            except AssertionError: return -np.inf

            # I seem to get a lot of "SVD did not converge" messages for
            # lower-dimension datasets. Might be something I should look
            # into, for now I'm rejecting these points with negative
            # infinite likelihood
            except np.linalg.linalg.LinAlgError: return -np.inf

        # Create the pymc model
        self.model = pm.Model([mu, cov, skew, data])


    def _calc_map(self, run=True):
        """ Find the point of optimum log-likelihood. """
        if run: 
            map_ = pm.MAP(self.model)
            map_.fit(method='fmin_powell')
        
            # Store the optimum point
            self.map_point = {
                'mean' : np.array(self.model.get_node('mu').value),
                'cov'  : np.array(self.model.get_node('cov').value),
                'skew' : np.array(self.model.get_node('skew').value),
            }

            with open(self.name + '_map.p', 'wb') as f:
                pickle.dump(self.map_point, f)

        else:
            with open(self.name + '_map.p', 'rb') as f:
                self.map_point = pickle.load(f)


    def sample(self, run=True):
        """ Run the PyMC algorithm to determine prior
        distributions for the estimated distribution parameters """

        if run:
            self.S = pm.MCMC(self.model, db='pickle',
                             dbname=self.name + '.p') 
            self.S.sample(iter=10000,burn=5000)
            self.S.commit()
        else:
            self.S = pm.database.pickle.load(self.name + '.p')

        self.avg_point = {
            'mean' : self.S.trace('mu')[:].mean(0),
            'cov'  : self.S.trace('cov')[:].mean(0),
            'skew' : self.S.trace('skew')[:].mean(0),
        }
            


    def run(self, run=True):
        """ Convienience function to run model """

        self._create_pymc_model()
        self._calc_map(run)
        self.sample(run)
        return self


    def plot_map(self, ax, i, j, pars=None):
        """ Plot the data points and likelyhood contours for data[:,i],
        data[:,j], based on the maximum a priori point """

        low = self.data.min(0)
        high = self.data.max(0)
        # low  = np.percentile(self.data, 1, axis=0)
        # high = np.percentile(self.data, 99, axis=0)
        res = 200

        if not pars: pars = self.map_point

        oned = [np.linspace(l, h, res) for l,h in zip(low, high)]
        twod = np.meshgrid(oned[i], oned[j])

        vals = np.ones((res**2, self.p))*self.data.mean(0)
        vals[:,i] = twod[0].flatten()
        vals[:,j] = twod[1].flatten()

        Z = np.exp(logmvsn(vals, pars['mean'], pars['cov'],
                           pars['skew']).reshape(twod[0].shape))

        # levels = np.array([0.07, 0.05, 0.03, 0.01])
        # levels = np.array([ 0.02,  0.04,  0.06,  0.08,  0.1 ,  0.12])

        blue = '#9999ff'
        ax.plot(self.data[:,i], self.data[:,j], '.', alpha=1.0,
                zorder=0, color=blue, ms=1.0)
        c = ax.contour(twod[0], twod[1], Z, colors='k',
                       zorder=1, linewidths=0.5)


    def plot_full(self, pars=None):
        """ Function to plot the correlation matrix of each variable's
        interaction """

        fig, axmatrix = plt.subplots(nrows=self.p, ncols=self.p,
                                     sharex='col', sharey='row',
                                     figsize=(3.5, 3.5))

        # Plot the scatter plot in each off-diagonal element, put the
        # name of the data in each diagonal element
        for i in xrange(self.p):
            for j in xrange(self.p):
                if i != j: self.plot_map(axmatrix[i,j], j, i, pars)
                else:
                    axmatrix[i,j].text(
                        0.5, 0.5, self.labels[i],
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axmatrix[i,j].transAxes)
                    axmatrix[i,j].set_axis_off()






def direct_from_centered(mean, cov, skew):
    """ Follow the advice of Arellano-Valle, R. B. & Azzalini, A. The
    centred parametrization for the multivariate skew-normal
    distribution. J. Multivar. Anal. 99, 1362-1382 (2008). 

    Given centered parameters, return direct parameters
    """

    # Calculate mu_z from Eq. 5
    c = cbrt((2*skew)/(4 - np.pi))
    u_z = c / np.sqrt(1 + c**2)

    # Calculate delta_z
    b = np.sqrt(2./np.pi)
    delta_z = u_z / b

    sigma_z = np.diag(np.sqrt(1 - (b*delta_z)**2))
    sigma = np.diag(np.sqrt(np.diag(cov)))

    w = sigma.dot(np.linalg.pinv(sigma_z))
    wdotu_z = w.dot(u_z)
    dp_mean = mean - wdotu_z
    dp_cov = cov + np.outer(wdotu_z, u_z).dot(w)
    winv = np.linalg.pinv(w)
    omega_hat = winv.dot(dp_cov).dot(winv)
    omega_hat_inv = np.linalg.pinv(omega_hat)
    ohi_dot_delta = omega_hat_inv.dot(delta_z)
    dp_skew = ohi_dot_delta/np.sqrt(1 - ohi_dot_delta.dot(delta_z))

    assert is_pos_def(dp_cov), "Covariance matrix must be pos-def"
    assert np.all(np.isfinite(dp_skew)), "Skew must be finite"

    return dp_mean, dp_cov, dp_skew


def _corr_from_cov(cov):
    """ Function to return a correlation matrix (1's on diagonal) from a
    covariance matrix """

    w = np.diag(np.sqrt(np.diag(cov)))
    winv = np.linalg.pinv(w)
    return winv.dot(cov).dot(winv), w

def _delta(corr, alpha):
    """ Function to calculate the delta parameter (skewness) from the
    correlation and alpha shape parameters. Based on Eq. 4 from
    doi:10.1111/1467-9868.00194 """

    alpha = np.atleast_2d(alpha).T
    return corr.dot(alpha)/np.sqrt(1 + alpha.T.dot(corr).dot(alpha))

def gen_mvsn_samples(size, mean, cov, skew):
    """ Sample a multivariate skewed normal distrubition, generating
    'size' number of samples. Formula from A. Azzalini, A. Capitanio, J.
    R. Stat. Soc. Ser. B (Statistical Methodol. 61, 579-602 (1999). 
    Uses centered parameters for the input arguments """
    
    p = len(mean)

    dp_mean, dp_cov, dp_skew = direct_from_centered(mean, cov, skew)

    corr, w = _corr_from_cov(dp_cov)
    delta = _delta(corr, dp_skew)
    corr_star = np.bmat([[np.array([[1]]) , delta.T] ,
                         [delta           , corr]])

    X = np.random.multivariate_normal(np.zeros(p+1), corr_star, size)
    neg = X[:,0] <= 0
    X[neg, 1:] *= -1
    return w.dot(X[:, 1:].T).T + dp_mean

def is_pos_def(mat):
    """ Check the matrix mat for positive-definiteness. Returns False if
    the matrix fails the cholesky decomposition """
    try: np.linalg.cholesky(mat)
    except np.linalg.linalg.LinAlgError: return False
    return True

def logmvsn(y, mean, cov, skew):
    """ Evaluate the likelyhood of the point y in the multivariate skew
    normal parameterized by centered parameters:
        location index mean (p x 1)
        positive definite scale matrix cov (p x p)
        skewness vector skew (p x 1), assuming zero non-diagonal entries
    returns the likelyhood of point y (n x p).

    Uses the definition of a skewed normal distribution from Eq. 1 in 
    T. I. Lin, J. Multivar. Anal. 100, 257-265 (2009). 
    """

    y = np.atleast_2d(y)

    const = np.log(0.15915494309127426) # Numerically calculated

    dp_mean, dp_cov, dp_skew = direct_from_centered(mean, cov, skew)

    sig0 = lambda x: np.log(1. + erf(x/np.sqrt(2)))
    y0 = y - dp_mean
    w = np.diag(np.sqrt(np.diag(dp_cov)))
    eta = np.linalg.pinv(w).dot(dp_skew)

    # Somewhat confusing here, its the outer product on a row-by-row
    # basis. S0[i] (pxp) is the outer product of y0[i] (px1)
    S0 = np.einsum('...i,...j->...ij', y0, y0)

    dp_cov_inv = np.linalg.pinv(dp_cov)
    inv_cov_dot_s0 = dp_cov_inv.dot(S0).swapaxes(1, 0)

    t1 = -0.5*np.log(np.linalg.det(dp_cov))
    t2 = -0.5*np.trace(inv_cov_dot_s0, axis1=1, axis2=2)
    t3 = sig0(eta.dot(y0.T))
    
    return const + t1 + t2 + t3


if __name__ == "__main__":

    true_cov = np.array([[1.   , 0.5  , 0.25] ,
                         [0.5  , 2.   , 0.35] ,
                         [0.25 , 0.35 , 3.]])
    true_mean = np.array([1., 0., 2.]) 
    true_skew = np.array([-0.1, 0.0, 0.1])

    size = 1000
    vals = gen_mvsn_samples(size=size, mean=true_mean, cov=true_cov,
                            skew=true_skew)


    se = SkewEstimator(vals, name='test').run(run=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

