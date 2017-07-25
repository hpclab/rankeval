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


def plot_feature_importance(feature_importance, feature_count,
                            max_features=10, sort_by="gain"):
    """
    Shows the most important features as a bar plot.

    Parameters
    ----------
    feature_importance : numpy.array
        Feature importance stats of the model to be visualized
    feature_count : numpy.array
        Feature count stats of the model to be visualized
    max_features : int or None
        Maximul number of features to be visualized. If None is passed, it will
        show all the features
    sort_by : 'gain' or 'count'
        The method to use for selecting the top features to display. 'gain'
        method selects the top features by importance, 'count' selects the top
        features by usage (i.e., number of times it has been used by a split
        node).

    Returns
    -------
    : matplotlib.figure.Figure
        The matpotlib Figure
    """

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

    opacity = 0.6
    error_config = {'ecolor': '0.3'}

    bar1 = ax1.bar(index, top_importances, bar_width,
                   alpha=opacity,
                   color='r',
                   error_kw=error_config,
                   align='center',
                   zorder=5)
    bar2 = ax2.bar(index + bar_width, top_counts, bar_width,
                   alpha=opacity,
                   color='b',
                   error_kw=error_config,
                   align='center',
                   zorder=5)

    ax1.set_title('Top-k Features by %s' % title_by)

    ax1.set_xlabel("Features")
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(top_features + 1)

    ax1.set_xlim(-bar_width/2 - bar_width, max_features - 1 + bar_width*5/2)

    align_y_axis(ax1, ax2, 0.05, 100)

    ax1.set_ylabel("Importance Gain")
    ax2.set_ylabel("Usage Count ")

    ax1.grid(True, ls='--', zorder=0)
    ax2.grid(False)

    legend = ax1.legend((bar1, bar2), ("Importance", "Count"),
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