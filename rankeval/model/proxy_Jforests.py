# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Class providing the implementation for loading/storing a QuickRank model
from/to file.

The Jforests project is described here: https://github.com/yasserg/jforests

The Jforests format adopts an XML representation. There is an ensemble node,
with a sub-node for each tree, identified by the "Tree" tag, followed by the
description of the tree (with splitting and leaf nodes). The splitting nodes are
described with two information: the feature-id used for splitting, and the
threshold value. Leaf nodes on the other hand are described by a "LeafOutputs"
tag with the value as content.
"""

from .rt_ensemble import RTEnsemble

try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree


class ProxyJforests(object):
    """
    Class providing the implementation for loading/storing a Jforests model
    from/to file.
    """

    @staticmethod
    def load(file_path, model):
        """
        Load the model from the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved
        model : RTEnsemble
            The model instance to fill
        """
        n_trees, n_nodes = ProxyJforests._count_nodes(file_path)
        # Initialize the model and allocate the needed space
        # given the shape and size of the ensemble
        model.initialize(n_trees, n_nodes)

        # get an iterable
        context = etree.iterparse(file_path, events=("start", "end"))

        # get the root element
        _, root = next(context)

        curr_tree = -1
        root_node = 0
        num_leaves = num_splits = 0

        for event, elem in context:

            if event == 'start' and elem.tag == 'Tree':
                    curr_tree += 1  # increase the current number index
                    root_node += num_leaves + num_splits
                    # save the curr node as the root of a new tree
                    model.trees_root[curr_tree] = root_node
                    model.trees_weight[curr_tree] = elem.attrib['weight']

            if event == 'end':

                if elem.tag == 'SplitFeatures':
                    split_features = map(int, elem.text.split(" "))
                    num_splits = 0
                    for pos, feature in enumerate(split_features):
                        num_splits += 1
                        model.trees_nodes_feature[root_node + pos] = feature
                elif elem.tag == 'LeftChildren':
                    left_children = map(int, elem.text.split(" "))
                    for pos, child in enumerate(left_children):
                        if child >= 0:
                            model.trees_left_child[root_node + pos] = \
                                root_node + child
                        else:
                            model.trees_left_child[root_node + pos] = \
                                root_node + num_splits + abs(child) - 1
                elif elem.tag == 'RightChildren':
                    right_children = map(int, elem.text.split(" "))
                    for pos, child in enumerate(right_children):
                        if child >= 0:
                            model.trees_right_child[root_node + pos] = \
                                root_node + child
                        else:
                            model.trees_right_child[root_node + pos] = \
                                root_node + num_splits + abs(child) - 1
                elif elem.tag == 'OriginalThresholds':
                    thresholds = map(float, elem.text.split(" "))
                    for pos, threshold in enumerate(thresholds):
                        model.trees_nodes_value[root_node + pos] = threshold
                elif elem.tag == 'LeafOutputs':
                    leaf_values = map(float, elem.text.split(" "))
                    num_leaves = 0
                    for pos, leaf_value in enumerate(leaf_values):
                        num_leaves += 1
                        model.trees_nodes_value[root_node + num_splits + pos] \
                            = leaf_value

            # clear the memory
            if event == 'end':
                elem.clear()    # discard the element
                root.clear()    # remove child reference from the root

    @staticmethod
    def save(file_path, model):
        """
        Save the model onto the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has to be saved
        model : RTEnsemble
            The model RTEnsemble model to save on file

        Returns
        -------
        status : bool
            Returns true if the save is successful, false otherwise
        """
        raise NotImplementedError("Feature not implemented!")

    @staticmethod
    def _count_nodes(file_path):
        """
        Count the total number of nodes (both split and leaf nodes)
        in the model identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved

        Returns
        -------
        tuple(n_trees, n_nodes) : tuple(int, int)
            The total number of trees and nodes (both split and leaf nodes)
            in the model identified by file_path.
        """
        # get an iterable
        # NOTE: it seems like there is a bug inside lxmx since selecting only
        # terminal tags with events=("end",) some tags are skipped...
        context = etree.iterparse(file_path, events=("start", "end"))

        # get the root element
        _, root = next(context)

        n_nodes = 0
        n_trees = 0
        for event, elem in context:
            if event != "end":
                continue
            if elem.tag == 'Tree':
                n_trees += 1
            elif elem.tag == 'SplitFeatures' or elem.tag == 'LeafOutputs':
                n_nodes += len(elem.text.split(" "))

            elem.clear()    # discard the element
            root.clear()    # remove root reference to the child

        return n_trees, n_nodes
