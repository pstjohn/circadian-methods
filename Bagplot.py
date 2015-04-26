import numpy as np
import rpy2.robjects as robjects

from matplotlib.patches import Polygon

from CommonFiles.PlotOptions import lighten_color

def calculate_bagplot(x, y, **kwargs):
    """ calculate_bagplot calculates bagplot parameters based on the
    method specificed by P. J. Rousseeuw, I. Ruts, J. W. Tukey (1999).
    The bagplot: a bivariate boxplot, The American Statistician,
    vol. 53, no. 4, 382-387.  This function is a wrapper around the
    R function bagplot provided by the aplpack library. 

    Parameters
    ----------

    x : 1-D sequence of floats
       x is the x-coordinate of the data
    y : 1-D sequence of floats
       y is the y-coordinate of the data

    **kwargs specify additional arguments to be passed directly to the
    boxplot function, overriding defaults.

    'factor'          : 2.5,
    'create_plot'     : False,
    'approx_limit'    : 300,
    'show_outlier'    : True,
    'show_looppoints' : True,
    'show_bagpoints'  : True,
    'dkmethod'        : 2,
    'show_whiskers'   : False,
    'show_loophull'   : True,
    'show_baghull'    : True,
    'verbose'         : False

    Returns
    -------
    dict, containing outputs from the R bagplot object

    """
    # Load APLPACK, library which contains the bagplot function
    robjects.r("library(aplpack)")

    # Import bagplot into the python namespace
    bagplot = robjects.r['bagplot']
    
    # Create an R matrix from the python arrays.
    v = robjects.FloatVector(np.hstack([x, y]))
    m = robjects.r['matrix'](v, ncol=2)

    bag_kwargs = {
        'factor'          : 2.5,
        'create_plot'     : False,
        'approx_limit'    : 300,
        'show_outlier'    : True,
        'show_looppoints' : True,
        'show_bagpoints'  : True,
        'dkmethod'        : 2,
        'show_whiskers'   : False,
        'show_loophull'   : True,
        'show_baghull'    : True,
        'verbose'         : False
    }

    bag_kwargs.update(kwargs)

    # Calculate bagplot data
    bag_data = bagplot(m, **bag_kwargs)

    # Process output results into python data models
    bag_dict = {
        'center'      : np.array(bag_data[0]),
        'hull_center' : np.array(bag_data[1]),
        'hull_bag'    : np.array(bag_data[2]),
        'hull_loop'   : np.array(bag_data[3]),
        'pxy_bag'     : np.array(bag_data[4]),
        'pxy_outer'   : np.array(bag_data[5]),
        'pxy_outlier' : np.array(bag_data[6]),
        'hdepths'     : np.array(bag_data[7]),
        'is_one_dim'  : bool(bag_data[8]),
        'xydata'      : np.array(bag_data[-1]),
    }

    pr_list = bag_data[9]
    pr_dict = {
        'sdev'     : np.array(pr_list[0]),
        'rotation' : np.array(pr_list[1]),
        'center'   : np.array(pr_list[2]),
        'scale'    : bool(pr_list[3]),
        'x'        : np.array(pr_list[4]),
    }
        
    bag_dict['pr_data'] = pr_dict

    return bag_dict


def bagplot(ax, x, y, color, hc=False, filled='bag'):

    bag_dict = calculate_bagplot(x, y)

    hc_poly = Polygon(bag_dict['hull_center'], 
                      fc=lighten_color(color, 0.25),
                      ec='none', zorder=0)
    hb_poly = Polygon(bag_dict['hull_bag'], 
                      fc=lighten_color(color, 0.5),
                      ec='none', zorder=0)
    hl_poly = Polygon(bag_dict['hull_loop'], 
                      fc=lighten_color(color, 0.75),
                      ec='none', zorder=0)

    hc_poly_l = Polygon(bag_dict['hull_center'], ec=color, fc='none',
                      zorder=1, ls='solid')
    hb_poly_l = Polygon(bag_dict['hull_bag'], ec=color, fc='none',
                      zorder=1, ls='solid')
    hl_poly_l = Polygon(bag_dict['hull_loop'], ec=color, fc='none',
                      zorder=1, ls='dashed')

    center = bag_dict['center']

    if filled == 'bag':
        ax.add_patch(hb_poly)
        if hc: ax.add_patch(hc_poly)

    elif filled:
        ax.add_patch(hl_poly)
        ax.add_patch(hb_poly)
        if hc: ax.add_patch(hc_poly)

    ax.add_patch(hl_poly_l)
    ax.add_patch(hb_poly_l)
    if hc: ax.add_patch(hc_poly_l)
    ax.plot(center[0], center[1], 'o', color=color)

