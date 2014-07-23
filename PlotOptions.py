# Bunch of overrides for matplotlib settings not changeable in
# matplotlibrc. To use, 
#
# >>> from CommonFiles.PlotOptions import PlotOptions, layout_pad
# >>> PlotOptions()
# >>> *** make plots here ***
# >>> fig.tight_layout(**layout_pad)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from CommonFiles.mimic_alpha import colorAlpha_to_rgb


def PlotOptions(uselatex=False):

    import matplotlib
    import matplotlib.axis, matplotlib.scale 
    from matplotlib.ticker import (MaxNLocator, NullLocator,
                                   NullFormatter, ScalarFormatter)

    MaxNLocator.default_params['nbins']=6
    MaxNLocator.default_params['steps']=[1, 2, 5, 10]

    def set_my_locators_and_formatters(self, axis):
        # choose the default locator and additional parameters
        if isinstance(axis, matplotlib.axis.XAxis):
            axis.set_major_locator(MaxNLocator())
        elif isinstance(axis, matplotlib.axis.YAxis):
            axis.set_major_locator(MaxNLocator())
        # copy&paste from the original method
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    #override original method
    matplotlib.scale.LinearScale.set_default_locators_and_formatters = \
            set_my_locators_and_formatters

    matplotlib.backend_bases.GraphicsContextBase.dashd = {
            'solid': (None, None),
            'dashed': (0, (2.0, 2.0)),
            'dashdot': (0, (1.5, 2.5, 0.5, 2.5)),
            'dotted': (0, (0.25, 1.50)),
        }

    matplotlib.colors.ColorConverter.colors['f'] = \
            (0.3058823529411765, 0.00784313725490196,
             0.7450980392156863)
    matplotlib.colors.ColorConverter.colors['h'] = \
            (0.5843137254901961, 0.0, 0.11372549019607843)
    matplotlib.colors.ColorConverter.colors['i'] = \
            (0.00784313725490196, 0.2549019607843137,
             0.0392156862745098)
    matplotlib.colors.ColorConverter.colors['j'] = \
            (0.7647058823529411, 0.34509803921568627, 0.0)

    if uselatex:
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='serif')
        

# Padding for formatting figures using tight_layout
layout_pad = {
    'pad'   : 0.05,
    'h_pad' : 0.6,
    'w_pad' : 0.6}

# Plot shortcuts for a number of circadian-relevant features

def plot_gray_zero(ax, **kwargs):
    ax.axhline(0, ls='--', color='grey', **kwargs)

def format_2pi_axis(ax, x=True, y=False):
    import numpy as np
    if x:
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xlim([0, 2*np.pi])
        ax.set_xticklabels(['$0$', r'$\nicefrac{\pi}{2}$', r'$\pi$',
                            r'$\nicefrac{3\pi}{2}$', r'$2\pi$'])
    if y:
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticklabels([r'$-\pi$', r'$-\nicefrac{\pi}{2}$','$0$',
                            r'$\nicefrac{\pi}{2}$', r'$\pi$'])

# def highlight_xrange(ax, xmin, xmax, color='y', alpha=0.5, **kwargs):
#     ax.axvspan(xmin, xmax, color=color, alpha=alpha, **kwargs)

class HistRCToggle:
    """ Class to toggle the xtick directional update of
    histogram-specific RC settings """

    hist_params = {'xtick.direction' : 'out',
                   'ytick.direction' : 'out'}

    def __init__(self):
        self.rcdef = plt.rcParams.copy()

    def on(self):
        plt.rcParams.update(self.hist_params)

    def off(self):
        plt.rcParams.update(self.rcdef)


blue = '#9999ff'
red = '#ff9999'

def histogram(ax, data1=None, data2=None, color1=blue, color2=red,
              bins=20, range=None, alpha=1., label1=None, label2=None):
    """ Function to display a pretty histogram of up to two different
    distributions. Approximates transparency to get around annoying
    issue of pdflatex and matplotlib. """
    
    weights1 = np.ones_like(data1)/len(data1)

    hist1 = ax.hist(data1, range=range, bins=bins, weights=weights1,
                    facecolor=color1, edgecolor='white', label=label1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('axes', -0.05))
    if range:
        ax.set_xlim([range[0]*(1-1E-3), range[1]*(1+1E-3)])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    if data2 is not None:
        weights2 = np.ones_like(data2)/len(data2)
        c2_on_w = colorAlpha_to_rgb(color2, alpha=0.5, bg='w')[0]
        c2_on_c1 = colorAlpha_to_rgb(color2, alpha=0.5, bg=color1)[0]

        hist2 = ax.hist(data2, range=range, bins=bins, weights=weights2,
                        facecolor=c2_on_w, edgecolor='white',
                        label=label2)

        ax.legend(loc='upper left')

        orders = hist2[0] > hist1[0]
        for i, order in enumerate(orders):
            if order:
                hist1[-1][i].set_facecolor(c2_on_c1)
                hist1[-1][i].set_zorder(2)
            else:
                hist2[-1][i].set_facecolor(c2_on_c1)
        return (hist1, hist2)

    else: 
        ax.legend(loc='upper left')
        return hist1


def density_contour(ax, xdata, ydata, nbins_x, nbins_y, range=None,
                    levels=None, **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a
        new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata,
                                       bins=(nbins_x, nbins_y),
                                       normed=True, range=range)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    contour = ax.contour(X, Y, Z, levels=levels, colors='0.2')
    ax.contourf(X, Y, Z, levels=levels, cmap=matplotlib.cm.PuBu,
                **contour_kwargs)

    return contour

def lighten_color(color, degree):
    cin = matplotlib.colors.colorConverter.to_rgb(color)
    cw = np.array([1.0]*3)
    return tuple(cin + (cw - cin)*degree)

def color_range(NUM_COLORS, cm=None):
    if cm is None: cm = matplotlib.cm.get_cmap('gist_rainbow')
    return (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))

