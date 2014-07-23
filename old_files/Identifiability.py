import casadi as cs
import numpy  as np
import matplotlib

from CommonFiles.Collocation import Collocation

from Utilities import (lighten_color, checksensid, find_labels)
from Utilities import colors as colors_default

whiskertol = 1E-2

class Identifiability:
    """
    Class to assist in running bootstrap approaches to parameter and
    sensitivitiy identifiability using the Collocation class. Should
    provide easy plotting methods and pdf export for use on the cluster.
    Hopefully will automatically generate simulated datapoints based on
    input data and standard deviations. Also will hopefully someday
    switch to a "measurement" based data system, where data is specified
    using linear functions of the state variables.
    """

    def __init__(self, model, name='Identifiability_Test'):
        """
        Data input should proceed state-by-state to allow for uneven
        sampling. model input should be a standard casadi SXFunction
        model, eventually/potentially generated through some form of
        genetic algorithm?
        """

        self.name = name
        self.model = model
        self.NEQ = model.input(cs.DAE_X).size()
        self.NP  = model.input(cs.DAE_P).size()
        self.confidence_level = 0.95

        self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.NEQ)]
        self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.NP)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.NP)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.NEQ)):
            self.ydict[par] = ind



        # Data to be passed to the Collocation instances, so that it is
        # accessable from outside the class (default values here)
        self.CollOpt = {
            'TF'            : 24.,
            'NK'            : 20,
            'max_iter'      : 4000,
            'max_cpu_time'  : 20*60,
            'FPgaurd'       : False,
            'PARMAX'        : np.array([1E+1]*self.NP),
            'PARMIN'        : np.array([1E-3]*self.NP),
            'XMAX'          : 1E+2,
            'XMIN'          : 1E-3,
            'PreParamEst'   : True,
            'weights'       : True, # Use Errors in objective function
            'print_level'   : 2,
            'stability'     : False,
            'p_init'        : None,
            'x_init'        : None,
            'state_weights' : np.ones(self.NEQ)
        }

        # Data to be passed to Collocation's BVP methods. Entries here
        # will overwrite defaults in Collocation.
        self.BvpOpts = {}



    def MeasurementMatrix(self, matrix):
        """
        Attaches the matrix A which specifies the measured output states. 
        y(t) = A x(t), where x are the internal states.
        """
        height, width = matrix.shape
        assert width == self.NEQ, "Measurement Matrix shape mis-match"
        self.NM = height
        self.Amatrix = matrix

        self.ydata = self.NM*[[]]
        self.tdata = self.NM*[[]]
        self.edata = self.NM*[[]]



    def AttachMeasurementData(self, measurement, times, values,
                              errors=False):
        """
        Attach each measurements's data points

        Requirements: AttachModel
        """
        
        # If error not provided, set to ones)
        if errors is False: errors = np.ones(len(times))

        # Check appropriate sizes
        assert(len(times) == len(values) == len(errors))

        # Make sure the measurement index is appropriate
        assert(self.NM > measurement >= 0)
        
       
        # Make sure data is within t-bounds. (this might want to throw
        # an error as well)
        
        times = np.array(times)
        if np.any(times >= self.CollOpt['TF']):
            if np.any(times > self.CollOpt['TF'] + 1E-3):
                print "BOUNDS WARNING: measurement in measurement %s" \
                        %measurement,
                print " has time above TF" 
            times = times%self.CollOpt['TF']

        # make sure data is in acending order.
        sort_inds = times.argsort()

        if type(values) is np.ndarray: values = values[sort_inds].tolist()
        if type(times)  is np.ndarray: times  = times[sort_inds].tolist()
        if type(errors) is np.ndarray: errors = errors[sort_inds].tolist()

        self.ydata[measurement] = values
        self.tdata[measurement] = times
        self.edata[measurement] = errors


    def CreateSeedData(self, repetitions, method='random_normal'):
        """
        Use standard deviations in self.edata to generate (repetitions)
        number of bootstrap data sets. Method will choose between
        different types of parameter space sampling.

        methods (should be): uniform should go from -1 stdev -> +1 stdev
        random_normal
        random_uniform
        sobol_normal
        sobol_uniform
        """

        assert repetitions > 0, "Number of Repetitions must be > 0"

        self.repetitions = repetitions

        # Here we will create a dictionary of random numbers depending
        # on the algorithm used. Will get multiplied by self.edata, so
        # these numbers should be mean 0, dev 1 normally distributed
        # numbers or [-1,1] range of uniform data
        
        # bootstrap_data[measure_function][repetitions][time_index]
        self.bootstrap_data = [[]] * self.NM


        from CommonFiles.sobol import i4_sobol
        self.sobol_seed = 0

        def get_sobol_uniform(datalen, repetitions):
            """
            function to generate sobol sequences on (0,1)
            """

            sobol_sequence = []
            for i in xrange(repetitions):
                array_out, self.sobol_seed = i4_sobol(datalen,
                                                      self.sobol_seed)
                sobol_sequence += [array_out]
            return np.array(sobol_sequence)

        def get_sobol_normal(datalen, repetitions):
            """
            Use Box Muller to convert uniform sobol sequence into
            appropriate uniform gaussian distribution
            """

            def box_muller(x1, x2):
                """
                uses two uniform numbers on (0,1) to generate two
                normal distributed values
                """
                z1 = np.sqrt(-2. * np.log(x1))*np.cos(2 * np.pi * x2)
                z2 = np.sqrt(-2. * np.log(x1))*np.sin(2 * np.pi * x2)

                return(z1, z2)



            rand_len = datalen
            if datalen%2: rand_len += 1
                
            sobol_sequence = get_sobol_uniform(rand_len, repetitions)
            normal = box_muller(sobol_sequence[:,:(rand_len/2)], 
                                sobol_sequence[:,(rand_len/2):])
            sobol_normal = np.hstack(normal)[:,:datalen]
            return sobol_normal


        for mm in xrange(self.NM):
            datalen = len(self.edata[mm])

            random_numbers = {
                'random_normal'  : np.random.standard_normal((datalen,
                                     repetitions)).T,
                'random_uniform' : np.random.rand(datalen,
                                     repetitions).T*2. - 1.,
                'sobol_uniform'  : get_sobol_uniform(datalen,
                                     repetitions)*2. - 1.,
                'sobol_normal'   : get_sobol_normal(datalen,
                                     repetitions)
            }

            self.bootstrap_data[mm] = (np.array(self.ydata[mm])
                    * np.ones((datalen,repetitions)).T
                    + np.array(self.edata[mm])
                    * random_numbers[method])

    def CollocationSolve(self, repetition_number):
        """
        Set up and solve the collocation problem for a given repetition
        number. These calls should be parallelized through a dtm.map
        function (or similar). The options for the collocation class
        should likely be accessable from outside the class, perhaps I'll
        change this later

        This function creates and return a sensitivity class with key
        properties already calculated. This function must return a
        pickleable object, else the parallel mapping will fail.
        """
        
        assert 0 <= repetition_number <= self.repetitions, \
               "Out of Bounds"

        collclass = Collocation(self.model)
        collclass.TF = self.CollOpt['TF']
        collclass.NK = self.CollOpt['NK']

        collclass.NLPdata['ObjMethod'] = 'lsq'
        collclass.NLPdata['FPgaurd'] = self.CollOpt['FPgaurd']
        collclass.NLPdata['stability'] = self.CollOpt['stability']
        collclass.NLPdata['state_weights'] = self.CollOpt['state_weights']

        collclass.IpoptOpts['max_iter'] = self.CollOpt['max_iter']
        collclass.IpoptOpts['max_cpu_time'] = self.CollOpt[
            'max_cpu_time']
        collclass.IpoptOpts['print_level'] = self.CollOpt['print_level']
        # collclass.IpoptOpts['tol'] = 1E-6

        # Set bounds.
        collclass.PARMAX = self.CollOpt['PARMAX']
        collclass.PARMIN = self.CollOpt['PARMIN']
        collclass.XMAX = self.CollOpt['XMAX']
        collclass.XMIN = self.CollOpt['XMIN']

        # Overwrite defaults in BVP options
        for key, value in self.BvpOpts.iteritems():
            collclass.BvpOpts[key] = value

        collclass.CoefficentSetup()
        
        collclass.MeasurementMatrix(self.Amatrix)

        # Input this bootstrap data to the Collocation class
        for i in xrange(0, self.NM):
            points = self.bootstrap_data[i][repetition_number].tolist()
            if self.CollOpt['weights']:
                collclass.AttachMeasurementData(i, self.tdata[i],
                                        points, errors=self.edata[i])
            else:
                collclass.AttachMeasurementData(i, self.tdata[i],
                                points)


        try:
            collclass.SetInitialValues(x_init=self.CollOpt['x_init'],
                                       p_init=self.CollOpt['p_init'])
        except Exception as ex: 
            # Quietly print error
            print ex
            return [str(ex)]

        collclass.ReformModel()
        collclass.Initialize() 
        collclass.CollocationSetup()

        try:
            collclass.CollocationSolve()
            collclass.bvp_solution()
            print "BVP solution successful"
            collclass.limitcycle.first_order_sensitivity()
            collclass.limitcycle.remove_unpickleable_objects()
            collclass.limitcycle.x_opt = collclass.NLPdata['x_opt']
            collclass.limitcycle.tgrid = collclass.tgrid

        except Exception as ex: 
            # Quietly print error
            print ex
            return [str(ex)]

        return collclass.limitcycle

    def ProcessResults(self, results):
        """
        Function to handle the return from a mapped call to
        CollocationSolve. Should break down the results into useable
        pieces for plotting functions and further analysis.

        result classes will be cleaned of all casadi and pBase
        attributes, so most function calls will likely be broken. These
        should instead be performed in CollocationSolve.
        """
        
        # Split returned results into error and convergent lists
        self.results = []
        self.errors = []
        for entry in list(results):
            if type(entry) is list:
                # Errors returned in lists
                self.errors += [entry]
            else:
                self.results += [entry]


        # Set up storage structures in current class to contain arrays
        # with the solution information from each
        # sensitivity/collocation class.

        self.data_list = ['bootstrap_data', 'parameters', 'ts', 'sols',
                          'periodsens', 'paramlabels', 'statelabels',
                          'x_opt', 'tgrid']

        for entry in self.data_list[1:]: # Not bootstrap_data
            assert not hasattr(self, entry), entry + " already present"
            setattr(self,entry,[])

        for base in self.results:
            self.ts += [base.ts]
            self.parameters += [base.paramset]
            self.sols += [base.sol]
            self.periodsens += [base.reldTdp]
            self.tgrid += [base.tgrid]
            self.x_opt += [base.x_opt]


        # See error types
        print ""
        print "COLLOCATION ERRORS:"
        print "-------------------"
        print "(" + repr(len(self.errors)) + "/",
        print repr(self.repetitions) + ') errors'
        for entry in self.errors: print entry
            
        assert len(self.results) > 2, "Not Enough Convergent Results"

        # Create arrays
        self.ts = np.array(self.ts)
        self.sols = np.array(self.sols)
        self.parameters = np.array(self.parameters)
        self.periodsens = np.array(self.periodsens)
        self.tgrid = np.array(self.tgrid)
        self.x_opt = np.array(self.x_opt)
        
        # We only need one
        self.paramlabels = np.array(self.results[0].plabels)
        self.statelabels = np.array(self.results[0].ylabels)


    def save_data(self):
        """
        This function should provide some means of saving the data to a
        picklable file, such that it can be loaded later (load_data?). 
        """
        import cPickle
        savefile = open(self.name + '.p', "wb")
        save_dict = {}

        # Save appropriate results
        for entry in self.data_list:
            assert hasattr(self,entry), entry + " missing"
            save_dict[entry] = getattr(self, entry)

        cPickle.dump(save_dict, savefile, protocol=-1)

    def load_data(self):
        """
        Opposite of save_data, this function should take data stored in
        a pickled object and re-allow the use of the plotting functions
        declared in this class.
        """
        import cPickle
        savefile = open(self.name + '.p', "rb")
        save_dict = cPickle.load(savefile)

        for key, value in save_dict.iteritems():
            setattr(self, key, value)

    def state_lookup(self, state_str):
        """
        returns the index of the state corresponding to the input
        state_str. Must be called after ProcessResults or load_data.
        """

        if state_str not in self.statelabels: 
            raise ValueError('State \''
                             + state_str + '\' not found')

        for i, entry in enumerate(self.statelabels):
            if state_str == entry: return i

    def param_lookup(self, param_str):
        """
        returns the index of the parameter corresponding to the input
        param_str. Must be called after ProcessResults or load_data.
        """

        if param_str not in self.paramlabels: 
            raise ValueError('Parameter \''
                             + param_str + '\' not found')

        for i, entry in enumerate(self.paramlabels):
            if param_str == entry: return i



    def IdentifiabilityPlot(self, states=None, output='pdf',
                            plots=['ts', 'pars', 'sens']):
        """
        Main function to plot default values for identifiability
        analysis. Should be the only function that needs import
        matplotlib, with subsequent plot functions accepting
        pre-determined axes as input arguments

        output = "pdf" for PDF file (cluster), or "disp" for matplotlib
        default.

        plots = which figures to create. default is all plots
        """

        if output is 'pdf':
            matplotlib.use('Agg')
            from matplotlib.backends.backend_pdf import PdfPages
            pdf = PdfPages(self.name + '.pdf')

        import matplotlib.pylab as plt

        if not states:
            if self.NM > 3: states = range(4)
            else: states = range(self.NM)

        percent_convergent = 100*(float(len(self.sols))/self.repetitions)
        convergent_string = str(percent_convergent) + ' % Convergent'

        if 'ts' in plots:
            figts = plt.figure()
            axts = figts.add_subplot(111)
            self.plot_ts_lines(axts, states)
            self.plot_data_boxes(axts, states)
            axts.text(0.95, 0.95, convergent_string,
                      horizontalalignment='right',
                      verticalalignment='top', transform =
                      axts.transAxes) 
            if output is 'pdf': pdf.savefig(transparent=True)

        if 'pars' in plots:
            figpars = plt.figure()
            axpars = figpars.add_subplot(111)
            self.plot_par_whisker(axpars)
            if output is 'pdf': pdf.savefig(transparent=True)

        if 'sens' in plots:
            figsens = plt.figure()
            axsens = figsens.add_subplot(111)
            self.plot_sens_whisker(axsens)
            if output is 'pdf': pdf.savefig(transparent=True)

        if output is 'pdf': pdf.close()
        elif output is 'disp': plt.show()

    def plot_ts_lines(self, ax, states, rasterize=True,
                      colors=colors_default, entryfilter=None):
        """
        Jack's idea of plotting each solution with a low alpha
        """

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(6))

        if entryfilter is None:
            entryfilter = np.array([True]*len(self.parameters))

        sols = self.sols[entryfilter]
        ts = self.ts[entryfilter]

        # Rasterize all elements below zorder 1
        if rasterize: ax.set_rasterization_zorder(1)
        ax.set_xlim(0.0, self.CollOpt['TF'])

        # print alpha
        alpha = 0.5

        for i, ind in enumerate(states):
            ax.plot(ts[:200].T,
                    self.Amatrix[ind].dot(sols[:200].swapaxes(2,1)).T,
                    color=colors[i], lw=0.5, alpha=alpha,
                    rasterized=rasterize, zorder=0)


    def plot_ts_shade(self, ax, states, colors=colors_default,
                      entryfilter=None, res=100):
        """
        Shade regions of possible dynamic responses
        """

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(6))

        if entryfilter is None:
            entryfilter = np.array([True]*len(self.parameters))

        sols = self.sols[entryfilter]
        ts = self.ts[entryfilter]

        skip = int(len(self.ts[0])/res)


        for i, ind in enumerate(states):
            y = self.Amatrix[ind].dot(sols[:200,::skip,:].swapaxes(2,1)).T
            ax.fill_between(ts[0,::skip], np.percentile(y, 90., axis=1),
                            np.percentile(y, 10., axis=1),
                            facecolor=lighten_color(colors[i],.6),
                            interpolate=True, color=lighten_color(colors[i],.6),
                            linestyle='-', lw=0)


    def plot_ts_gaussian(self, ax, states, yres=200, tres=200):
        """
        Function to plot gaussian distribution of dynamic responses.
        """

        from scipy.stats import gaussian_kde
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(6))

        # Graph Boundarys
        states = np.array(states)
        ymin = self.Amatrix[states].dot(self.sols.swapaxes(2,1)).min()
        ymax = self.Amatrix[states].dot(self.sols.swapaxes(2,1)).max()
        tmin = 0.
        tmax = self.CollOpt['TF']

        # resolution
        tskip = len(self.ts[0])/tres
        yrange = np.linspace(ymin, ymax, yres)

        for i, ind in enumerate(states):
            # Condense time and dynamic response to managable size
            sol = self.Amatrix[ind](self.sols[:,::tskip].T).T
  
            density = []

            for entry in sol.T:
                k = gaussian_kde(entry)
                density += [k(yrange)]

            density = np.array(density)
            maxdensity = density[0].sum()
            density = density/maxdensity

            colorin = \
            matplotlib.colors.colorConverter.to_rgba(colors_default[i])

            # Create colormap
            cmap = LinearSegmentedColormap.from_list('my_cmap',
                    [colorin, colorin], 1024)
            cmap._init()
            alphas = np.linspace(0, 1., cmap.N+3)
            cmap._lut[:,-1] = alphas

            vscale = 0.05*(yres)/200.

            ax.imshow(np.rot90(density), aspect='auto', cmap=cmap,
                      extent=[tmin, tmax, ymin, ymax], vmax=vscale,
                      interpolation='bicubic')


    def plot_data_boxes(self, ax, states, colors=colors_default,
                        yscale=False):
        """
        Plot data confidence intervals

        :param ax: matplotlib axes instance
        :param list states: list of states to plot
        :param list colors: list of colors to use
        :param bool yscale: whether or not the boxplot should scale
                            y-axis
        """
        import matplotlib.pylab as plt

        # Boxplot screws up the xticks, so we'll save the incoming
        # xticks and restore them after the call to boxplot()
        # We also keep the ylimits to be the limits of the shaded
        # regions
        xticks = ax.get_xticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for i,ind in enumerate(states):
            bp = ax.boxplot(self.bootstrap_data[ind], widths=1, 
                            positions=self.tdata[ind], sym='',
                            patch_artist=True)

            plt.setp(bp['medians'], color=colors[i], linewidth=1.3,
                     zorder=3)
            plt.setp(bp['boxes'], color=colors[i],
                     facecolor=lighten_color(colors[i],.4),
                     linewidth=1.3, zorder=2)
            plt.setp(bp['whiskers'], color=colors[i], linewidth=1.3)
            plt.setp(bp['caps'], color=colors[i], linewidth=1.3)


        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        if not yscale: ax.set_ylim(ylim)
        else:
            # Protect against negative bounds
            ylim = ax.get_ylim()
            if ylim[0] < 0.: ax.set_ylim([0, ylim[1]])


    def plot_par_whisker(self, ax, params=None, entryfilter=None):
        """
        Logarithmic plot of parameter distributions to demonstrate
        parameter identifiability

        params = list of parameter indexes to plot. If None, plots all
        parameters

        entryfilter = a numpy boolean array that only plots a subset of
        the returned solution
        """

        if params is None: params = range(self.NP)
        if entryfilter is None:
            entryfilter = np.array([True]*len(self.parameters))

        NP = len(params)
        EF = entryfilter

        from scipy.stats import gaussian_kde
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pylab as plt

        w = min(0.15*max(NP, 1.0), 0.5)
        
        ax.yaxis.set_major_locator(MaxNLocator(5))
        bp = ax.boxplot(self.parameters[EF][:,params],
                        sym='',patch_artist=True,
                        widths=w)
        plt.setp(bp['medians'], color='black', linewidth=1.0)
        plt.setp(bp['boxes'], color='black', facecolor='#E5E5E5',
                 linewidth=1.)
        plt.setp(bp['whiskers'], color='black', linewidth=1.0)
        plt.setp(bp['caps'], color='black', linewidth=1)
        

        for p,d in enumerate(self.parameters[EF][:,params].T):
            try:
                p += 1
                #calculates the kernel density
                k = gaussian_kde(np.log10(d))
                # use parameter max/mins
                #
                PARMAX = np.log10(self.CollOpt['PARMAX'])
                PARMIN = np.log10(self.CollOpt['PARMIN'])
                try:
                    m = max(PARMIN, k.dataset.min())
                    M = min(PARMAX, k.dataset.max())
                except ValueError:
                    m = max(PARMIN.min(), k.dataset.min())
                    M = min(PARMAX.max(), k.dataset.max())

                # m = k.dataset.min() #lower bound of violin
                # M = k.dataset.max() #upper bound of violin
                # x = np.logspace(np.log10(m), np.log10(M), base=10,
                #                 num=100)
                x = np.arange(m,M,(M-m)/100.) # support for violin
                v = k.evaluate(x) #violin profile (density curve)
                #scaling the violin to the available space
                v = v/v.max()*w 

                ax.fill_betweenx(10**x, -v+p, v+p, where=(v > whiskertol),
                                 facecolor=lighten_color('g', 0.7),
                                 lw=0)

            except (ValueError, np.linalg.LinAlgError): pass

        ax.set_yscale('log')
        ax.set_xlim([0.5, NP+0.5])
        if NP <= 6:
            ax.set_xticks(range(1, NP+1))
            ax.set_xticklabels(self.paramlabels[params])
        elif 6 < NP < 20:
            ax.set_xticks(range(1, NP+1))
            ax.set_xticklabels(self.paramlabels[params], rotation=90)
        else:
            ax.set_xticks(find_labels(NP))
                


    def plot_sens_whisker(self, ax, params=None, entryfilter=None,
                          bounds=None, labels=None):
        """
        Plot function to plot distributions in senstivity values,
        shading senstivities which fail a 95% identifiable test as red.
        See manuscript (in prep)

        :param list params: parameter indexes to plot.
                            If None, plots all parameters
        :param np.ndarray entryfilter: a numpy boolean array that only
                                       plots a subset of the returned
                                       solution
        :param tuple bounds: (ymin, ymax), bounds for the sensitivity plots
        :param list labels: parameter labels for the plot. Only used if
                            len(par) < 20.
        """

        if params is None: params = np.array(range(self.NP))
        else: params = np.array(params)

        if entryfilter is None:
            entryfilter = np.array([True]*len(self.parameters))

        if labels is None: labels = self.paramlabels[params]
        else: assert len(labels) == len(params), "Label length error"

        NP = len(params)
        EF = entryfilter
        
        from scipy.stats import gaussian_kde
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pylab as plt

        ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.set_yticks([-1.5,-0.8,0.0,0.8,1.5])
        ids = checksensid(self.periodsens[EF][:,params],
                          level=self.confidence_level)
        
        # Dividing line for sens = 0
        ax.plot([0,NP+0.5],[0, 0],'k--')

        # Identifiable
        w = min(0.15*max(NP,1.0),0.5)
        
        if np.any(ids):
            bp = ax.boxplot(self.periodsens[EF][:,params[ids]],
                            positions=np.array(range(1,NP+1))[ids],
                            sym='', patch_artist=True, widths=w)
            plt.setp(bp['medians'], color='black', linewidth=1.0)
            plt.setp(bp['boxes'], color='black', facecolor='#E5E5E5',
                     linewidth=1.0)
            plt.setp(bp['whiskers'], color='black', linewidth=1.0)
            plt.setp(bp['caps'], color='black', linewidth=1.0)

        # Non-identifiable
        if not np.all(ids):
            bp = ax.boxplot(self.periodsens[EF][:,params[~ids]],
                            positions=np.array(range(1,NP+1))[~ids],
                            sym='', patch_artist=True, widths=w)
            plt.setp(bp['medians'], color='red', linewidth=1.0)
            plt.setp(bp['boxes'], color='red',
                     facecolor=lighten_color('red', 0.7), linewidth=1.0)
            plt.setp(bp['whiskers'], color='red', linewidth=1.0)
            plt.setp(bp['caps'], color='red', linewidth=1.0)
            

        # Get appropriate bounds for sensitivities
        if bounds is None:
            medians = abs(np.median(self.periodsens[EF][:, params], axis=0))
            medians.sort()
            end = int(NP/4)
            ymax = round(6 * medians[-end], 1)/2.
            ymin = -ymax
        else:
            ymin = bounds[0]
            ymax = bounds[1]

        for p,d in enumerate(self.periodsens[EF][:, params].T):
            try:
                
                # trim density to plot boundaries
                def trim(d):
                    return (d > ymin) & (d < ymax)
                d = filter(trim, d)

                p += 1
                #calculates the kernel density
                k = gaussian_kde(d, bw_method='silverman')
               
                # Use graph bounds for appropriate max/mins
                m = max(ymin, k.dataset.min())
                M = min(ymax, k.dataset.max())
                # m = k.dataset.min() #lower bound of violin
                # M = k.dataset.max() #upper bound of violin

                x = np.arange(m,M,(M-m)/100.) # support for violin
                v = k.evaluate(x) #violin profile (density curve)
                #scaling the violin to the available space
                v = v/v.max()*w 
                if ids[p - 1]:
                    ax.fill_betweenx(x, -v+p, v+p, where=(v > whiskertol),
                                     facecolor=lighten_color('g', 0.7),
                                     lw=0)
                else:
                    ax.fill_betweenx(x, -v+p, v+p, where=(v > whiskertol),
                                     facecolor=lighten_color('r', 0.7),
                                     lw=0)

            except (ValueError, np.linalg.LinAlgError): pass

        ax.set_ylim([ymin, ymax])
        ax.set_xlim([0.5, NP+0.5])

        # Handle tick labels

        if NP <= 7:
            # Long labels?
            if max([len(label) for label in labels]) > 7:
                ax.set_xticks(range(1, NP+1))
                ax.set_xticklabels(labels, rotation=45)
            else:
                ax.set_xticks(range(1, NP+1))
                ax.set_xticklabels(labels)

        elif 7 < NP < 20:
            ax.set_xticks(range(1, NP+1))
            ax.set_xticklabels(labels, rotation=90)
            
        else:
            ax.set_xticks(find_labels(NP))



if __name__ == '__main__':
    from CommonFiles.pBase import pBase
    from CommonFiles.tyson2statemodel import model, y0in, paramset

    nk = 10
    maxrelerror = 0.15
    maxsyserror = 0.10

    np.random.seed(1)

    orig = pBase(model(), paramset, y0in)
    orig.limitCycle()

    ts = np.linspace(0, 24., nk, endpoint=False)
    sol = orig.lc(ts * y0in[-1] / 24.)

    A = np.eye(orig.NEQ)

    def get_random(shape):
        return (np.random.rand(*shape) - 0.5)

    y_exact = np.array([A.dot(sol[i]) for i in xrange(len(sol))]) 
    y_error = (y_exact * (1 + maxrelerror * get_random(y_exact.shape) +
                          maxsyserror * y_exact.max(0) *
                          get_random(y_exact.shape)))
    errors = (maxrelerror*y_exact + maxsyserror*y_exact.max(0))
    


    IdentifyClass = Identifiability(model())
    IdentifyClass.CollOpt['TF'] = 24.
    IdentifyClass.CollOpt['NK'] = 10
    IdentifyClass.CollOpt['PreParamEst'] = True
    IdentifyClass.CollOpt['PARMIN'] = np.array(paramset)/10.
    IdentifyClass.CollOpt['PARMAX'] = np.array(paramset)*10.
    # IdentifyClass.CollOpt['stability'] = 10.
    # IdentifyClass.CollOpt['max_cpu_time'] = 0.1

    A = np.eye(IdentifyClass.NEQ)
    IdentifyClass.MeasurementMatrix(A)

    for i in xrange(IdentifyClass.NM):
        IdentifyClass.AttachMeasurementData(i, ts, y_exact[:,i], errors[:,i])
     

    nruns = 15
    IdentifyClass.CreateSeedData(nruns, method='random_uniform')
    

    results = map(IdentifyClass.CollocationSolve, 
                            xrange(nruns))

    IdentifyClass.ProcessResults(results)
    import pylab as plt
    # fig, axmatrix = plt.subplots(nrows=2, ncols=1)
    # IdentifyClass.plot_par_whisker(axmatrix[0], params=range(5))
    # IdentifyClass.plot_sens_whisker(axmatrix[1], params=range(5))
    # IdentifyClass.plot_ts_lines(ax, [0,1], rasterize=False)
    # IdentifyClass.plot_data_boxes(ax, [0,1])
    # plt.show()
    IdentifyClass.IdentifiabilityPlot()
    plt.show()

    # IdentifyClass.save_data()
