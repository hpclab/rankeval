"""
This package provides support for feature analysis visualizations.
"""

from __future__ import print_function
import six

import matplotlib.pyplot as plt

import numpy as np

try:
    xrange
except NameError:
    # Python3's range is Python2's xrange
    xrange = range


def plot_feature_importance(feature_perf, max_features=10, sort_by="gain",
                            feature_names=None):
    """
    Shows the most important features as a bar plot.

    Parameters
    ----------
    feature_perf : xarray.DataArray
        Feature importance stats of the model to be visualized
    max_features : int or None
        Maximul number of features to be visualized. If None is passed, it will
        show all the features
    sort_by : 'gain' or 'count'
        The method to use for selecting the top features to display. 'gain'
        method selects the top features by importance, 'count' selects the top
        features by usage (i.e., number of times it has been used by a split
        node).
    feature_names : list of string
        The name of the features to use for plotting. If None, their index is
        used in place of the name (starting from 1).

    Returns
    -------
    : matplotlib.figure.Figure
        The matpotlib Figure
    """

    feature_importance = feature_perf.sel(type='importance').data
    feature_count = feature_perf.sel(type='count').data.astype(np.uint16)

    # figure
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax2 = ax1.twinx()

    if sort_by == "gain":
        idx_sorted = np.argsort(feature_importance)[::-1]
        title_by = "Importance"
    elif sort_by == "count":
        idx_sorted = np.argsort(feature_count)[::-1]
        title_by = "Count"
    else:
        raise RuntimeError("Sorting of features for visualization "
                           "not supported!")

    if isinstance(max_features, six.integer_types):
        idx_sorted = idx_sorted[:max_features]
    else:
        max_features = len(feature_importance)

    top_features = idx_sorted
    top_importances = feature_importance[idx_sorted]
    top_counts = feature_count[idx_sorted]

    index = np.arange(max_features)
    bar_width = 0.35

    opacity = 0.7

    bar1 = ax1.bar(index, top_importances, bar_width,
                   alpha=opacity,
                   color='r',
                   align='center',
                   zorder=5,
                   edgecolor='black')
    bar2 = ax2.bar(index + bar_width, top_counts, bar_width,
                   alpha=opacity,
                   color='b',
                   align='center',
                   zorder=5,
                   edgecolor='black')

    ax1.set_title('Top-k Features by %s' % title_by)

    ax1.set_xlabel("Features")
    if feature_names is not None:
        feature_names_f = np.array(["%16s" % f for f in feature_names])
        ax1.set_xticks(index + bar_width / 2 + 0.15)
        ax1.set_xticklabels(feature_names_f[idx_sorted], rotation=45,
                            ha="right")
    else:
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels(top_features + 1)

    ax1.set_xlim(-bar_width/2 - bar_width, max_features - 1 + bar_width*5/2)

    step_y = np.ceil(top_importances.max() * 10) / 100
    align_y_axis(ax1, ax2, step_y, 100, num_ticks=6)

    ax1.set_ylabel("Importance Gain")
    ax2.set_ylabel("Usage Count")

    ax1.grid(False)
    ax2.grid(False)
    ax1.yaxis.grid(True, ls='--', zorder=0)

    ax1.legend((bar1, bar2), ("Importance", "Count"),
               loc='best', shadow=True, frameon=True, fancybox=True)

    return fig


def align_y_axis(ax1, ax2, minresax1, minresax2, num_ticks=7):
    """ Sets tick marks of twinx axes to line up with num_ticks total tick marks

    ax1 and ax2 are matplotlib axes
    Spacing between tick marks will be a factor of minresax1 and minresax2"""

    ax1ylims = ax1.get_ybound()
    ax2ylims = ax2.get_ybound()
    ax1factor = minresax1 * (num_ticks - 1)
    ax2factor = minresax2 * (num_ticks - 1)
    ax1.set_yticks(np.linspace(ax1ylims[0],
                               ax1ylims[1]+(ax1factor -
                               (ax1ylims[1]-ax1ylims[0]) % ax1factor) %
                               ax1factor,
                               num_ticks))
    ax2.set_yticks(np.linspace(ax2ylims[0],
                               ax2ylims[1]+(ax2factor -
                               (ax2ylims[1]-ax2ylims[0]) % ax2factor) %
                               ax2factor,
                               num_ticks))