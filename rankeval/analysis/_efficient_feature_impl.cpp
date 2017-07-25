#include "_efficient_feature_impl.h"

#include <math.h>
#include <vector>
#include <iostream>

void c_feature_importance(
        const float* X,
        const float* y,
        const int* trees_root,
        const float* trees_weight,
        const short* trees_nodes__feature,
        const float* trees_nodes_value,
        const int* trees_left_child,
        const int* trees_right_child,
        float* feature_imp,
        short* feature_count,
        int n_instances,
        int n_features,
        int n_trees) {

    // initialize features importance
    #pragma omp parallel for
    for (unsigned int feature = 0; feature < n_features; ++feature) {
        feature_imp[feature] = 0;
    }

    // default scores on the root node of the first tree
    std::vector<float> y_pred(n_instances, 0);
    std::vector<float> y_pred_tree(n_instances);

    for (unsigned int tree_id=0; tree_id<n_trees; ++tree_id) {
        c_feature_importance_tree(X,
                                  y,
                                  trees_root,
                                  trees_weight,
                                  trees_nodes__feature,
                                  trees_nodes_value,
                                  trees_left_child,
                                  trees_right_child,
                                  tree_id,
                                  feature_imp,
                                  feature_count,
                                  n_instances,
                                  n_features,
                                  y_pred.data(),
                                  y_pred_tree.data());
    }
}

void c_feature_importance_tree(
        const float* X,
        const float* y,
        const int* trees_root,
        const float* trees_weight,
        const short* trees_nodes_feature,
        const float* trees_nodes_value,
        const int* trees_left_child,
        const int* trees_right_child,
        const int tree_id,
        float* feature_imp,
        short* feature_count,
        const int n_instances,
        const int n_features,
        float* y_pred,
        float* y_pred_tree) {

    // Initialize the vector of instance ids used to identify which instances
    // reaches each tree node (the split node partition this set)
    std::vector<unsigned int> split_instance(n_instances);
    // The residual scores to fit
    // y_target = y - y_pred
    std::vector<float> y_target(n_instances);
    float mean_y_target = 0;
    #pragma omp parallel for reduction( + : mean_y_target )
    for (unsigned int instance = 0; instance < n_instances; ++instance) {
        split_instance[instance] = instance;
        y_target[instance] = y[instance] - y_pred[instance];
        mean_y_target += y_target[instance];
    }
    mean_y_target /= n_instances;

    // initialize the y_pred_tree vector
    // y_pred_tree = np.full(n_instances, fill_value=y_target.mean())
    #pragma omp parallel for
    for (unsigned int instance = 0; instance < n_instances; ++instance)
        y_pred_tree[instance] = mean_y_target;

    TreeNode root(trees_root[tree_id], 0, n_instances - 1);
    std::vector<TreeNode> queue = { root };

    while (!queue.empty()) {

        TreeNode node = queue.back();
        queue.pop_back();

        int node_id = node.node_id;
        short feature_id = trees_nodes_feature[node_id];
        float threshold = trees_nodes_value[node_id];

        feature_count[feature_id]++;

        // Split the instances in left-right (end_id will be the frontier)
        int start_id = node.start_id;
        int end_id = node.end_id;
        float y_target_mean_left = 0, y_target_mean_right = 0;
        unsigned int instance;
        while (start_id <= end_id) {
            instance = split_instance[start_id];
            if (X[instance * n_features + feature_id] <= threshold) {
                y_target_mean_left += y_target[instance];
                ++start_id;
            } else {
                y_target_mean_right += y_target[instance];
                std::swap(split_instance[start_id], split_instance[end_id]);
                --end_id;
            }
        }

        int left_docs = end_id - node.start_id + 1;
        int right_docs = node.end_id - end_id;

        // we need to normalize the mean y_targets (left and right)
        if (left_docs > 0)
            y_target_mean_left /= left_docs;
        if (right_docs > 0)
            y_target_mean_right /= right_docs;

        // compute split gain
        float delta_mse = 0;
        #pragma omp parallel for reduction( + : delta_mse )
        for (unsigned int i = node.start_id; i <= node.end_id; ++i) {
            unsigned int instance = split_instance[i];
            float pre_split_mse =
                pow(y_target[instance] - y_pred_tree[instance], 2);

            if (i <= end_id)
                y_pred_tree[instance] = y_target_mean_left;
            else
                y_pred_tree[instance] = y_target_mean_right;

            float post_split_mse =
                pow(y_target[instance] - y_pred_tree[instance], 2);

            delta_mse += pre_split_mse - post_split_mse;
        }

        // update feature importance
        feature_imp[feature_id] += delta_mse / n_instances;

        // if children are not leaves, add in the queue of the nodes to visit
        if (!is_leaf_node(trees_left_child[node_id],
                          trees_left_child,
                          trees_right_child) && end_id > node.start_id) {

            TreeNode left(trees_left_child[node_id], node.start_id, end_id);
            queue.push_back(left);
        }

        if (!is_leaf_node(trees_right_child[node_id],
                          trees_left_child,
                          trees_right_child) && node.end_id > (end_id + 1) ) {

            TreeNode right(trees_right_child[node_id], end_id + 1, node.end_id);
            queue.push_back(right);
        }
    }

    #pragma omp parallel for
    for (unsigned int instance = 0; instance < n_instances; ++instance) {
        y_pred_tree[instance] *= trees_weight[tree_id];
        y_pred[instance] += y_pred_tree[instance];
    }
}