"""Class providing the implementation for loading/storing a QuickRank model from/to file.

The QuickRank project is described here:
    http://quickrank.isti.cnr.it

The QuickRank format adopts an XML representation. There is an header section, identified by the "info" tag, with the
most important parameters adopted to learn such a model. It follows then the description of the ensemble, with a node
for each tree, identified by the "tree" tag, followed by the description of the tree (with splitting and leaf nodes).
The splitting nodes are described with two information: the feature id used for splitting, and the threshold value. Leaf
nodes on the other hand are described by an "output" tag with the value as content.
"""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>

from rankeval.core.model.proxy_model import ProxyModel
from rankeval.core.model.regression_trees_ensemble import RegressionTreesEnsemble

try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree


class ProxyQuickRank(ProxyModel):
    """
    Class providing the implementation for loading/storing a QuickRank model from/to file.
    """

    @staticmethod
    def load(file_path):
        """
        Load the model from the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved

        Returns
        -------
        model : RegressionTreesEnsemble
            The loaded model as a RegressionTreesEnsemble object
        """
        n_trees, n_nodes = ProxyQuickRank._count_nodes(file_path)
        # create the model and allocate the needed space
        model = RegressionTreesEnsemble(n_trees, n_nodes)

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
                    model.trees_root[curr_tree] = curr_node    # save the curr node as the root of a new tree
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
                    model.trees_nodes_feature[curr_node] = int(elem.text.strip()) - 1
                elif elem.tag == 'threshold' or elem.tag == 'output':
                    model.trees_nodes_value[curr_node] = elem.text.strip()

            # clear the memory
            if event == 'end':
                elem.clear()    # discard the element
                root.clear()    # remove child reference from the root
        return model

    @staticmethod
    def save(file_path):
        """
        Save the model onto the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has to be saved

        Returns
        -------
        status : bool
            Returns true if the save is successful, false otherwise
        """
        raise NotImplementedError("Feature not implemented!")

    @staticmethod
    def _count_nodes(file_path):
        """
        Count the total number of nodes (both split and leaf nodes) in the model identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved

        Returns
        -------
        tuple(n_trees, n_nodes) : tuple(int, int)
            The total number of trees and nodes (both split and leaf nodes) in the model identified by file_path.
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
