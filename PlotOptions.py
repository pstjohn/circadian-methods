# Bunch of overrides for matplotlib settings not changeable in
# matplotlibrc. To use, 
#
# >>> from CommonFiles.PlotOptions import PlotOptions, layout_pad
# >>> PlotOptions()
# >>> *** make plots here ***
# >>> fig.tight_layout(**layout_pad)

from CommonFiles.Utilities import lighten_color, color_range

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


