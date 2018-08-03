# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Class providing the implementation for loading/storing a QuickRank model
from/to file.

The QuickRank project is described here: http://quickrank.isti.cnr.it

The QuickRank format adopts an XML representation. There is an header section,
identified by the "info" tag, with the most important parameters adopted to
learn such a model. It follows then the description of the ensemble, with a node
for each tree, identified by the "tree" tag, followed by the description of the
tree (with splitting and leaf nodes). The splitting nodes are described with two
information: the feature id used for splitting, and the threshold value. Leaf
nodes on the other hand are described by an "output" tag with the value as
content.
"""

from six.moves import range

from .rt_ensemble import RTEnsemble

try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree


class ProxyQuickRank(object):
    """
    Class providing the implementation for loading/storing a QuickRank model
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
        n_trees, n_nodes = ProxyQuickRank._count_nodes(file_path)
        # Initialize the model and allocate the needed space
        # given the shape and size of the ensemble
        model.initialize(n_trees, n_nodes)

        # get an iterable
        context = etree.iterparse(file_path, events=("start", "end"))

        # get the root element
        _, root = next(context)

        curr_tree = curr_node = -1
        split_stack = []
        for event, elem in context:

            if event == 'start':
                if elem.tag == 'tree':
                    curr_tree += 1  # increase the current number index
                    curr_node += 1  # increase the current node index
                    # save the curr node as the root of a new tree
                    model.trees_root[curr_tree] = curr_node
                    model.trees_weight[curr_tree] = elem.attrib['weight']
                elif elem.tag == 'split':
                    if 'pos' in elem.attrib:
                        parent_node = split_stack[-1]
                        curr_node += 1
                        if elem.attrib['pos'] == 'left':
                            model.trees_left_child[parent_node] = curr_node
                        else:
                            model.trees_right_child[parent_node] = curr_node
                    split_stack.append(curr_node)
            else:   # event = 'end'
                if elem.tag == 'split':
                    split_stack.pop()
                elif elem.tag == 'feature':
                    model.trees_nodes_feature[curr_node] = \
                        int(elem.text.strip()) - 1
                elif elem.tag == 'threshold' or elem.tag == 'output':
                    model.trees_nodes_value[curr_node] = elem.text.strip()

            # clear the memory
            if event == 'end':
                elem.clear()    # discard the element
                root.clear()    # remove child reference from the root

    @staticmethod
    def _xmlprettyprint(elem, level=0, indent_str='\t'):
        i = "\n" + level * indent_str
        j = "\n" + (level - 1) * indent_str
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + indent_str
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                ProxyQuickRank._xmlprettyprint(child, level + 1, indent_str)
            # last child needs to have 1 indentation less in the following line
            elem.getchildren()[-1].tail = i
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

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

        # ROOT
        root = etree.Element("ranker")

        # Learning process info
        n_info = etree.SubElement(root, "info")

        etree.SubElement(n_info, "type").text = "MART"
        etree.SubElement(n_info, "trees").text = repr(model.n_trees)
        etree.SubElement(n_info, "leaves").text = repr(model.max_leaves())
        etree.SubElement(n_info, "shrinkage").text = repr(model.learning_rate)
        etree.SubElement(n_info, "leafsupport").text = repr(0)
        etree.SubElement(n_info, "discretization").text = repr(0)
        etree.SubElement(n_info, "estop").text = repr(0)

        # Ensemble
        n_ensemble = etree.Element("ensemble")
        root.append(n_ensemble)

        for idx_tree in range(model.n_trees):
            n_tree = etree.SubElement(
                n_ensemble, "tree",
                {
                    "id": repr(idx_tree+1),
                    "weight": repr(model.trees_weight[idx_tree])
                })
            cur_node = model.trees_root[idx_tree]

            n_split = ProxyQuickRank._get_node_xml_element(model, cur_node)
            n_tree.append(n_split)

        ProxyQuickRank._xmlprettyprint(root)

        with open(file_path, 'w') as f_out:
            f_out.write(etree.tostring(root).decode('utf-8'))

        return True

    @staticmethod
    def _get_node_xml_element(model, node_id, child=None):
        """
        Builds and xml.etree.ElementTree representing the given node (and its children).

        Parameters
        ----------
        model : RTEnsemble
            The model
        node_id : int
            The node id from which the visit is started
        child : string or None
            Whether the node is a "left" or "right" child. The None value identifies the root.

        Returns
        -------
        etree : xml.etree.ElementTree.Element
            The Element tree.
        """
        n_split = etree.Element("split")
        if child:
            n_split.set("pos", child)

        if model.is_leaf_node(node_id):
            etree.SubElement(n_split, "output").text = repr(model.trees_nodes_value[node_id])
        else:
            feature_idx = model.trees_nodes_feature[node_id] + 1
            feature_threshold = model.trees_nodes_value[node_id]
            etree.SubElement(n_split, "feature").text = repr(feature_idx)
            etree.SubElement(n_split, "threshold").text = repr(feature_threshold)

            child = ProxyQuickRank._get_node_xml_element(model, model.trees_left_child[node_id], "left")
            n_split.append(child)
            child = ProxyQuickRank._get_node_xml_element(model, model.trees_right_child[node_id], "right")
            n_split.append(child)

        return n_split

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
        context = etree.iterparse(file_path, events=("end",))

        # get the root element
        _, root = next(context)

        n_nodes = 0
        n_trees = 0
        for _, elem in context:
            if elem.tag == 'tree':
                n_trees += 1
            elif elem.tag == 'feature' or elem.tag == 'output':
                n_nodes += 1

            elem.clear()    # discard the element
            root.clear()    # remove root reference to the child

        return n_trees, n_nodes
