import numpy as np
import pandas as pd
from collections import deque

df = pd.read_csv('drug200.csv')
print(df)

data = df.to_numpy()
# print(data)
X = data[:150, :-1]
# print(X)
y = data[:150, -1]
# print(y)
X_type = np.array(([1, 0, 0, 0, 1]))
y_type = np.array(([0]))
# print(X_type, y_type)
X_test = data[:-50, :-1]
y_test = data[:-50, :-1]

class Decision_Tree:
    def __init__(self, max_features = np.inf, max_depth = None, min_samples = 2, avg_childs = 2, *, order = None):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.order = order
        self.nodes = 0
        #for continuous cases
        self.avg_childs = avg_childs
        
    def _pre_process(self, X, y, X_type, y_type):
        n_features = X.shape[1]
        features = None

        self.max_features = n_features
        if(n_features < 2 or X.shape[0] < self.min_samples):
            flag = False
            return n_features, flag
            
            
        # select the features to build the tree
        if(self.order is not None):
            features = [self.order[i] for i in range(self.max_features)]
        else:
            features = np.random.choice(np.arange(0, n_features), size = self.max_features, replace = False)

        flag = True
        return features, flag
        
    def build_tree(self, parent, X, y, X_type, y_type):
                
        #BFS method -- You can easily incorparate the max_depth if you have bfs knowledge
        q = deque()
        q.append([parent, X, y, X_type, y_type])
        while q:            
            child = q.popleft()
            #check exit condition
            features, status = self._check_exit_condition(child)
            if status == "leaf":
                print("Leafing the node")
                continue
            #find the best split for this node, update the node and q
            feature, threshold = self._select_best_feature(features, child[1] ,child[2], child[3], child[4])
            childs = []
            q, childs = self._update_q_and_childs(child, feature, threshold, q, childs)
            
            child[0].feature = feature
            child[0].feature_type = child[3][feature]
            child[0].dividers = threshold
            child[0].childs = childs
            
            self.nodes = self.nodes + 1
            self.avg_childs = self.avg_childs * self.nodes + len(childs)
            self.avg_childs = self.avg_childs / self.nodes

        return 
        
    def _check_exit_condition(self, child):
        
        status = "non_leaf"
        features, flag = self._pre_process(child[1], child[2], child[3], child[4])
        
        if not flag:
            child_value = None
            
            if child[4] == 0:
                #categorical
                values, counts = np.unique(child[2], return_counts = True)
                child_value = dict(zip(values, counts))
            else:
                #continuous
                child_value = child[2].mean(axis = 0)
                
            child[0].value = child_value
            print("leaf node with values", child[0].value)
            status = "leaf"

        return features, status
        
    def _update_q_and_childs(self, child, feature, threshold, q, childs):
        
        prev_idx = -np.inf #used in continuous case
        
        for idx in threshold:
            mask = None
            if child[3][feature] == 0:
                #categorical
                mask = child[1][:, feature] == idx
            else:
                #continuos
                mask = (child[1][:, feature] > prev_idx) & (child[1][:, feature] < idx)
                prev_idx = idx
                
            X_child = np.delete(child[1][mask, :], feature, axis = 1)
            y_child = child[2][mask]
            X_type_child = np.delete(child[3], feature)
            y_type_child = child[4]
            
            new_child = Node()
            
            childs.append(new_child)
            q.append([new_child, X_child, y_child, X_type_child, y_type_child])

        #Exiting the loop 
        if(child[3][feature] == 1):
            
            mask = child[1][:, feature] > prev_idx
            
            X_child = np.delete(child[1][mask, :], feature, axis = 1)
            y_child = child[2][mask]
            X_type_child = np.delete(child[3], feature)
            y_type_child = child[4]

            new_child = Node()
            childs.append(new_child)
            q.append([new_child, X_child, y_child, X_type_child, y_type_child])

        return q, childs

    def _select_best_feature(self, features, X, y, X_type, y_type):
        
        entropy_gain, variance_gain, Threshold = -np.inf, -np.inf, None
        ent_best_feature, var_best_feature = None, None
        curr_gain = None
        flag = None
        for feature in features:
            if(X_type[feature] == 0):
                #categorical
                threshold = np.unique(X[:, feature])
            else:
                #continuous
                threshold = self._find_threshold_continuous(feature, X, y, y_type)
            
            if(y_type):
                #con
                curr_gain = self._variance_gain(feature, X, y, threshold)
                if(variance_gain < curr_gain):
                    variance_gain = curr_gain
                    var_best_feature = feature
                    Threshold = threshold
            else :
                #cat
                curr_gain = self._entropy_gain(feature, X, y, threshold)
                if(entropy_gain < curr_gain):
                    entropy_gain = curr_gain
                    ent_best_feature = feature
                    Threshold = threshold
                
        if ent_best_feature is not None:
            return ent_best_feature, Threshold
        else:
            return var_best_feature, Threshold

    def _entropy_gain(self, feature, X, y, threshold):
        
        parent_gain = self._entropy(y)
        weighted_children = 0.0
        for idx in threshold:
            mask = X[:, feature] == idx
            y_child = y[mask]
            weighted_children = weighted_children + (len(y_child)/len(y)) * (self._entropy(y_child))
        gain = parent_gain - weighted_children
        
        return gain
    
    def _variance_gain(self, feature, X, y, threshold):
        parent_gain = self._variance(y)
        weighted_children = 0.0
        for idx in threshold:
            mask = X[:, feature] < idx # Caution: Here i am pre assuming that the threshold is divided such that it can be used in intervels
            y_child = np.array((y))[mask]
            weighted_children = weighted_children + (len(y_child)/len(y)) * (self._variance(y_child))
        gain = parent_gain - weighted_children
        return gain
        
    def _variance(self, y):
        return np.var(y) # Caution: Donot work for the other types than numericals

    def _entropy(self, y):
        n = len(y)
        if(n == 0):
            return 0.0
            
        values, count = np.unique(y, return_counts = True)
        entropy = 0.0
        for i in count:
            if(i != 0):
                entropy = entropy + (i/n) * np.log2(i/n)
        return -1 * entropy
    
    #Only for the continuous features
    def _find_threshold_continuous(self, feature, X, y, y_type):
        #if the label is continuos
        n_childs = self.avg_childs
        feature_map = np.stack((X[:, feature], y), axis = 1)
        feature_map = feature_map[feature_map[:, 1].argsort()]

        threshold = []
        if y_type == 1:
            #con label
            n = int(len(y) / n_childs)
            for i in range(n_childs):
                start = n * i
                end = n * (i + 1)
                if end >= (len(y)):
                    break
                threshold.append(feature_map[start : end, 0].mean(axis = 0))
        else:
            #cat label
            for i in range(len(feature_map) - 1):
                if(feature_map[i, 1] != feature_map[i+1, 1]):
                    threshold.append(feature_map[i, 0])
        return threshold
                    

class Node:
    def __init__(self, feature = None, feature_type = None, dividers = None, childs = None, *, value = None):
        self.feature = feature
        self.feature_type = feature_type
        self.dividers = dividers if dividers is not None else []
        self.childs = childs if childs is not None else []
        #Only for the leaf nodes
        self.value = value

    def depth(self):
        if(len(self.childs) == 0):
            return 0
        return 1 + max([child.depth() for child in self.childs])

    def predict(self, x):
        # shape of x is (1, n)
        if self.value is not None:
            return self.value

        if(self.feature_type == 0):
            #categorical
            print(self.dividers) #debug
            idx_value = x[self.feature]
            for i in range(len(self.dividers)):
                if(self.dividers[i] == idx_value):
                    return self.childs[i].predict(x)
            # If it comes out without matching anything
            print("Caution: The new value isn't in the data category so choosing a random child")
            i = np.random.randint(0, len(childs))
            return self.childs[i].predict(x)
        else:
            #continuous
            print(self.dividers) #debug
            idx_value = x[self.feature]
            for i in range(len(self.dividers)):
                if(self.dividers[i] < idx_value):
                    return self.childs[i].predict(x)
            # This is valid because the childs are 1 more than the dividers
            return self.childs[i].predict(x)
        
parent = Node()
d = Decision_Tree()
d.build_tree(parent, X, y, X_type, y_type)
X = np.array(([30, "MALE", "HIGH", "HIGH", 12]))
print(parent.predict(X))