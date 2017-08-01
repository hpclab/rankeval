"""
This package provides support for topological analysis visualizations. 
"""

from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, DictFormatter)

import numpy as np

from rankeval.analysis.topological import TopologicalAnalysisResult

try:
    xrange
except NameError:
    # Python3's range is Python2's xrange
    xrange = range


def plot_shape(topological, max_level=10):
    '''
    Shows the average tree shape as a bullseye plot.

    Parameters
    ----------
    topological : TopologicalAnalysisResult
        Topological stats of the model to be visualized.
    max_level : int
        Maximul tree-depth of the visualization. Maximum allowed value is 16.

    Returns
    -------
    : matplotlib.figure.Figure
        The matpotlib Figure
    '''

    ts = topological.avg_tree_shape()
    max_levels, _ = ts.shape
    max_levels = min(max_levels, max_level+1)
    # precision of the plot (should be at least 2**max_levels)
    max_nodes = max(128, 2**max_levels)

    # custom color map
    cm = LinearSegmentedColormap.from_list('w2r', [(1,1,1),(23./255.,118./255.,182./255.)], N=256)

    # figure
    fig = plt.figure(1, figsize=(12, 6))

    tr = Affine2D().translate(np.pi, 0) + PolarAxes.PolarTransform()
    x_ticks = [x+2. for x in range(max_levels-1)]
    grid_locator2 = FixedLocator( x_ticks )
    tick_formatter2 = DictFormatter({k:" " +str(int(k) -1) for k in x_ticks})
    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(0, np.pi, 0, max_levels),
                                                      grid_locator2=grid_locator2,
                                                      tick_formatter2 = tick_formatter2)

    ax3 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_axes(ax3)

    # fix axes
    ax3.axis["bottom"].set_visible(False)
    ax3.axis["top"].set_visible(False)
    ax3.axis["right"].set_axis_direction("top")
    
    ax3.axis["left"].set_axis_direction("bottom")
    ax3.axis["left"].major_ticklabels.set_pad(-5)
    ax3.axis["left"].major_ticklabels.set_rotation(180)
#    ax3.axis["left"].label.set_text("tree level")
#    ax3.axis["left"].label.set_ha("right")
    ax3.axis["left"].label.set_rotation(180)
    ax3.axis["left"].label.set_pad(20)
    ax3.axis["left"].major_ticklabels.set_ha("left")

    ax = ax3.get_aux_axes(tr)
    ax.patch = ax3.patch # for aux_ax to have a clip path as in ax
    ax3.patch.zorder=.9 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.            
    # data to be plotted
    theta, r = np.mgrid[0:np.pi:(max_nodes+1)*1j, 0:max_levels+1]
    z = np.zeros(theta.size).reshape(theta.shape)

    for level in xrange(max_levels):
        num_nodes = 2**level
        num_same_color = max_nodes/num_nodes
        for node in xrange(num_nodes):
            if node<ts.shape[1]:
                z[ num_same_color*node:num_same_color*(node+1), level ] = ts[level, node]

    # draw the tree nodes frequencies
    h = ax.pcolormesh(theta, r, z, cmap=cm, vmin=0., vmax=1.)

    # color bar
    cax = fig.add_axes([.95, 0.15, 0.025, 0.8]) # axes for the colot bar
    cbar = fig.colorbar(h, cax=cax, ticks=np.linspace(0,1,11))
    cbar.ax.tick_params(labelsize=8) 
    cax.set_title('Frequency', verticalalignment="bottom", fontsize=10)

    # separate depths
    for level in xrange(1,max_levels+1):
        ax.plot( np.linspace(0,np.pi,num=128), [1*level]*128, 'k:', lw=0.3)

    # separate left sub-tree from right sub-tree
    ax.plot( [np.pi,np.pi], [0,1], 'k-')
    ax.plot( [0,0], [0,1], 'k-')

    # label tree depths
    ax.text(np.pi/2*3,.2, "ROOT", horizontalalignment='center', verticalalignment='center')
    ax.text(-.08,max_levels, "Tree level", verticalalignment='bottom', fontsize=14)
    
    # left/right subtree
#    ax.text(np.pi/4.,max_levels*1.15, "Left subtree", fontsize=12,
#            horizontalalignment="center", verticalalignment="center")
#    ax.text(np.pi/4.*3,max_levels*1.15, "Right subtree", fontsize=12, 
#            horizontalalignment="center", verticalalignment="center")

    ax3.set_title("Average Tree Shape: " + str(topological.model), fontsize=16, y=1.2)

    return fig