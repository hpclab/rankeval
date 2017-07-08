class TreeNode {
    public:
        unsigned int node_id;
        unsigned int start_id;
        unsigned int end_id;

        TreeNode(unsigned int node_id,
                 unsigned int start_id,
                 unsigned int end_id) :
            node_id(node_id),
            start_id(start_id),
            end_id(end_id) {}

        int get_n_instances() {
            return end_id - start_id + 1;
        }
};

void c_feature_importance(
        const float* X,
        const float* y,
        const int* trees_root,
        const float* trees_weight,
        const short* trees_nodes_feature,
        const float* trees_nodes_value,
        const int* trees_left_child,
        const int* trees_right_child,
        float* feature_imp,
        short* feature_count,
        const int n_instances,
        const int n_features,
        const int n_trees);

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
        float* y_pred_tree);

inline bool is_leaf_node(int node_id,
                         const int* trees_left_child,
                         const int* trees_right_child) {
    return trees_left_child[node_id] == -1 && trees_right_child[node_id] == -1;
}