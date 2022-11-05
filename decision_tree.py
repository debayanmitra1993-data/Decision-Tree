class DecisionTree:
  def __init__(self, max_depth, min_samples_leaf):
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
  
  
  def helper(self, currnode, currdepth):
    # pure node criteria (Prune)
    if calculate_entropy(self.y_train[currnode.point_indices,0]) == 0:
      return
    
    # min samples leaf node criteria (Prune)
    if len(currnode.point_indices) < self.min_samples_leaf:
      return
    
    # max depth criteria (Prune)
    if currdepth > self.max_depth:
      return
    
    # Iterate across available features and get the best split point with highest IG
    maxig = float("-inf")
    for colidx in range(self.X_train.shape[1]):
      ig = calculate_information_gain(self.X_train[currnode.point_indices,colidx], self.y_train[currnode.point_indices,0])
      if ig > maxig:
        maxig = ig
        best_split_feature_idx = colidx
    
    # Generate Node & insert as child node of parent node
    print(self.X_train[currnode.point_indices,best_split_feature_idx].shape)
    for node_cat in label_counter(self.X_train[currnode.point_indices,best_split_feature_idx]).keys():
      node_indices = np.where(self.X_train[currnode.point_indices,best_split_feature_idx] == node_cat)
      generate_node = Node(node_indices)
      currnode.child_nodes.append(generate_node)
      self.helper(generate_node, currdepth + 1)
    return currnode
  
  def fit(self,X,y):
    self.X_train = X
    self.y_train = y
    self.root = Node(np.arange(self.X_train.shape[0]))
    self.decision_tree = self.helper(self.root, 0)
