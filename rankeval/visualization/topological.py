"""
This package provides support for topological analysis visualizations. 
"""

import numpy as np
import graphviz
from collections import defaultdict
import six

from numbers import Integral

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes

from mpl_toolkits.axisartist import floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, DictFormatter)

from rankeval.analysis.topological import TopologicalAnalysisResult
from rankeval.model import RTEnsemble

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
    cm = LinearSegmentedColormap.from_list(
        'w2r', [(1,1,1),(23./255.,118./255.,182./255.)], N=256
    )

    # figure
    fig = plt.figure(1, figsize=(12, 6))

    tr = Affine2D().translate(np.pi, 0) + PolarAxes.PolarTransform()
    x_ticks = [x+2. for x in range(max_levels-1)]
    grid_locator2 = FixedLocator( x_ticks )
    tick_formatter2 = DictFormatter({k:" " +str(int(k) -1) for k in x_ticks})
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, np.pi, 0, max_levels),
        grid_locator2=grid_locator2,
        tick_formatter2 = tick_formatter2
    )

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
        num_same_color = int(max_nodes/num_nodes)
        for node in xrange(num_nodes):
            start_idx = num_same_color * node
            end_idx = start_idx + num_same_color
            if node<ts.shape[1]:
                z[start_idx:end_idx, level] = ts[level, node]

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
    ax.text(
        np.pi/2*3,.2, "ROOT",
        horizontalalignment='center', verticalalignment='center'
    )
    ax.text(
        -.08,max_levels, "Tree level",
        verticalalignment='bottom', fontsize=14
    )

    # left/right subtree
#    ax.text(np.pi/4.,max_levels*1.15, "Left subtree", fontsize=12,
#            horizontalalignment="center", verticalalignment="center")
#    ax.text(np.pi/4.*3,max_levels*1.15, "Right subtree", fontsize=12,
#            horizontalalignment="center", verticalalignment="center")

    ax3.set_title(
        "Average Tree Shape: " + str(topological.model),
        fontsize=16, y=1.2)

    return fig


def export_graphviz(ensemble, tree_id=0, out_file=None, max_depth=None,
                    output_leaves=None, label='all', label_show=False,
                    feature_names=None, highlight_leaves=False,
                    leaves_parallel=False, node_ids=False, rounded=False,
                    proportion=False, special_characters=False, precision=3):
    """Export a single decision tree in DOT format.
    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::
        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)
    Parameters
    ----------
    ensemble : RTEnsemble of regression trees
        The ensemble of regression trees to be exported to GraphViz.
    tree_id : int (default 0)
        The tree identifier in the ensemble to export (starts from 0)
    out_file : file object or string, optional (default=None)
        Handle or name of the output file. If ``None``, the result is
        returned as a string.
    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.
    output_leaves : numpy 2d array (default=None
        Output leaves computed while scoring this model on a given dataset using
        the score method of the RTEnsemble model (the third return variable,
        y_leaves, reports the output leaf of each dataset sample for each tree).
    label : {'all', 'leaves'}, optional (default='all')
        Whether to compute the counts of dataset samples that during the scoring
        touched a specific node (i.e., how many times each node have been
        involved in the scoring activity, on a specific dataset).
        Options include 'all' to compute counts at every node or 'leaves' to
        compute it only on the leaf nodes.
    label_show : bool, optional (default=False)
        When set to ``True``, print counts information in each node.
    feature_names : list of strings, optional (default=None)
        Names of each of the features.
    highlight_leaves : bool, optional (default=False)
        When set to ``True``, paint leaves to indicate extremity of values for
        regression.
    leaves_parallel : bool, optional (default=False)
        When set to ``True``, draw all leaf nodes at the bottom of the tree.
    node_ids : bool, optional (default=False)
        When set to ``True``, show the ID number on each node.
    rounded : bool, optional (default=False)
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.
    proportion : bool, optional (default=False)
        When set to ``True``, change the display of 'counts'
        adding the percentages.
    special_characters : bool, optional (default=False)
        When set to ``False``, ignore special characters for PostScript
        compatibility.
    precision : int, optional (default=3)
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.
    Returns
    -------
    dot_data : string
        String representation of the input tree in GraphViz dot format.
        Only returned if ``out_file`` is None.
    """

    def get_color(value, colormap):
        hex_codes = ["%x" % i for i in np.arange(16)]
        cmap = plt.get_cmap(colormap)

        ratio = (value - colors['bounds'][0]) / \
                (colors['bounds'][1] - colors['bounds'][0])
        color = cmap(ratio)
        color = [int(np.round(c*255)) for c in color]
        # we increase the alpha in order to make the color brighter
        color[-1] = min(255, int(color[-1] * 0.75))
        color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]

        return '#' + ''.join(color)

    def node_to_str(ensemble, tree_id, node_id):
        # Generate the node content string

        # PostScript compatibility for special characters
        if special_characters:
            characters = [
                '&#35;', '<FONT POINT-SIZE="10"><SUB>',
                '</SUB></FONT>', '&le;', '<br/>', '>', '&#37;']
            node_string = '<'
        else:
            characters = ['#', '[', ']', '<=', '\\n', '"', '%']
            node_string = '"'

        # Write node ID
        if node_ids:
            node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        count_label = ""
        if output_leaves is not None and label_show:
            if label == 'all' or ensemble.is_leaf_node(node_id):
                proportion_txt = ""
                if proportion:
                    proportion_txt = " (%.2f%s)" % (
                        float(counts[node_id]) / colors['bounds'][1] * 100,
                        characters[6])
                count_label += 'counts = %s%s%s' % (counts[node_id],
                                                    proportion_txt,
                                                    characters[4])

        # Write decision criteria, except for leaves
        if not ensemble.is_leaf_node(node_id):
            feature_id = ensemble.trees_nodes_feature[node_id]
            feature_value = ensemble.trees_nodes_value[node_id]
            if feature_names is not None:
                feature_txt = feature_names[feature_id]
            else:
                feature_txt = "f%s%s%s " % (characters[1],
                                            feature_id,
                                            characters[2])
            if label_show:
                node_string += count_label
            node_string += '%s %s %s%s' % (feature_txt,
                                           characters[3],
                                           round(feature_value, precision),
                                           characters[4])
        else:
            # Write node output value, only for leaves
            value = ensemble.trees_nodes_value[node_id]
            value_text = np.around(value, precision)
            # Strip whitespace
            value_text = str(value_text.astype('S32')).replace("b'", "'")
            value_text = value_text.replace("' '", ", ").replace("'", "")
            value_text = value_text.replace("[", "").replace("]", "")
            value_text = value_text.replace("\n ", characters[4])
            if label_show:
                node_string += count_label
            node_string += 'output = ' + value_text + characters[4]


        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + characters[5]

    def recursive_count(ensemble, tree_id, counts, node_id):
        if not ensemble.is_leaf_node(node_id):
            left_count = recursive_count(ensemble, tree_id, counts,
                                         ensemble.trees_left_child[node_id])
            right_count = recursive_count(ensemble, tree_id, counts,
                                          ensemble.trees_right_child[node_id])
            counts[node_id] = left_count + right_count
        return counts[node_id]

    def recurse(ensemble, tree_id, node_id, parent_id=None, depth=0):
        if parent_id is not None and ensemble.is_leaf_node(parent_id):
            raise ValueError("Invalid node (parent %d is a LEAF) " % parent_id)

        # Add node with description
        if max_depth is None or depth <= max_depth:

            # Collect ranks for 'leaf' option in plot_options
            if ensemble.is_leaf_node(node_id):
                ranks['leaves'].append(str(node_id))
            else:
                ranks[str(depth)].append(str(node_id))

            out_file.write('%d [label=%s'
                           % (node_id,
                              node_to_str(ensemble, tree_id, node_id)))

            if output_leaves is not None:
                if ensemble.is_leaf_node(node_id) or label == 'all':
                    node_val = counts[node_id]
                    out_file.write(', fillcolor="%s"' %
                                   get_color(float(node_val),
                                             colormap='coolwarm'))
                else:
                    out_file.write(', fillcolor="%s"' % "#ffffffff")
            elif highlight_leaves:
                if ensemble.is_leaf_node(node_id):
                    node_val = ensemble.trees_nodes_value[node_id]
                    out_file.write(', fillcolor="%s"' %
                                   get_color(node_val,
                                             colormap='coolwarm'))
                else:
                    out_file.write(', fillcolor="%s"' % "#ffffffff")
            out_file.write('] ;\n')

            if parent_id is not None:
                # Add edge to parent
                out_file.write('%d -> %d' % (parent_id, node_id))
                if ensemble.trees_root[tree_id] == parent_id:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45])
                    out_file.write(' [labeldistance=2.5, labelangle=')
                    if ensemble.trees_left_child[parent_id] == node_id:
                        out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        out_file.write('%d, headlabel="False"]' % angles[1])
                out_file.write(' ;\n')

            if not ensemble.is_leaf_node(node_id):
                recurse(ensemble, tree_id, ensemble.trees_left_child[node_id],
                        parent_id=node_id,
                        depth=depth + 1)
                recurse(ensemble, tree_id, ensemble.trees_right_child[node_id],
                        parent_id=node_id,
                        depth=depth + 1)

        else:
            ranks['leaves'].append(str(node_id))
            out_file.write('%d [label="(...)"' % node_id)
            if highlight_leaves or output_leaves is not None:
                # color cropped nodes grey
                out_file.write(', fillcolor="#C0C0C0"')
            out_file.write('] ;\n' % node_id)

            if parent_id is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent_id, node_id))

    own_file = False
    return_string = False
    try:
        if isinstance(out_file, six.string_types):
            if six.PY3:
                out_file = open(out_file, 'w', encoding='utf-8')
            else:
                out_file = open(out_file, 'wb')
            own_file = True

        if out_file is None:
            return_string = True
            out_file = six.StringIO()

        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError("'precision' should be greater or equal to 0."
                                 " Got {} instead.".format(precision))
        else:
            raise ValueError("'precision' should be an integer. Got {}"
                             " instead.".format(type(precision)))

        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_ in the decision_tree
        if feature_names is not None:
            max_feature_id = np.max(ensemble.trees_nodes_feature) + 1
            if len(feature_names) > max_feature_id:
                raise ValueError("Length of feature_names, %d "
                                 "does not match number of features, %d"
                                 % (len(feature_names),
                                    max_feature_id))

        # Check tree_id is correct with reference the current ensemble
        if tree_id > ensemble.n_trees - 1:
            raise ValueError("Tree_id (%d) is higher than the number of trees "
                             "in the current ensemble (%d)" %
                             (tree_id, ensemble.n_trees))

        # The depth of each node for plotting with 'leaf' option
        ranks = defaultdict(list)
        # The colors to render each node with
        colors = dict()

        if output_leaves is not None:
            leaves_count = np.unique(output_leaves[:, tree_id],
                                     return_counts=True)
            counts = defaultdict((lambda:0), zip(*leaves_count))
            recursive_count(ensemble, tree_id, counts,
                            ensemble.trees_root[tree_id])

            if label == 'all':
                colors['bounds'] = (np.min(list(counts.values())),
                                    np.max(list(counts.values())))
            elif label == 'leaves':
                colors['bounds'] = (np.min(list(leaves_count[1])),
                                    np.max(list(leaves_count[1])))
            else:
                raise ValueError("Label parameter not supported, %s" % label)

        else:
            # Compute leaf output bounds to be used for colouring the leaves
            leaves_output = []
            start_id = ensemble.trees_root[tree_id]
            end_id = start_id + ensemble.num_nodes()[tree_id]
            for i in np.arange(start_id, end_id):
                if ensemble.is_leaf_node(i):
                    leaves_output.append(ensemble.trees_nodes_value[i])
            # Find max and min values in leaf nodes for regression
            colors['bounds'] = (np.min(leaves_output),
                                np.max(leaves_output))

        out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        out_file.write('node [shape=box')
        rounded_filled = []
        if highlight_leaves or output_leaves is not None:
            rounded_filled.append('filled')
        if rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            out_file.write(', style="%s", color="black"'
                           % ", ".join(rounded_filled))
        if rounded:
            out_file.write(', fontname=helvetica')
        out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if leaves_parallel:
            out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if rounded:
            out_file.write('edge [fontname=helvetica] ;\n')

        recurse(ensemble, tree_id, ensemble.trees_root[tree_id])

        # If required, draw leaf nodes at same depth as each other
        if leaves_parallel:
            for rank in sorted(ranks):
                out_file.write("{rank=same ; " +
                               "; ".join(r for r in ranks[rank]) + "} ;\n")
        out_file.write("}")

        if return_string:
            return out_file.getvalue()

    finally:
        if own_file:
            out_file.close()


def plot_tree(dot_data):
    return graphviz.Source(dot_data)