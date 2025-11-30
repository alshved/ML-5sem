import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    features_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    valid_splits = features_sorted[:-1] != features_sorted[1:]

    if not np.any(valid_splits):
        return np.array([]), np.array([]), None, None

    thresholds = (features_sorted[:-1][valid_splits] + features_sorted[1:][valid_splits]) / 2.0

    N = len(target_vector)
    total_1 = np.sum(target_vector)

    cum_sum = np.cumsum(target_sorted)

    left_1 = cum_sum[:-1][valid_splits]
    left_count = np.arange(1, N)[valid_splits]

    right_1 = total_1 - left_1
    right_count = N - left_count

    p1_l = left_1 / left_count
    p0_l = 1.0 - p1_l
    h_l = 1.0 - p1_l**2 - p0_l**2

    p1_r = right_1 / right_count
    p0_r = 1.0 - p1_r
    h_r = 1.0 - p1_r**2 - p0_r**2

    ginis = - (left_count / N) * h_l - (right_count / N) * h_r

    best_idx = np.argmax(ginis)

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split if min_samples_split is not None else 2
        self._min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else 1

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        N = len(sub_y)

        if np.all(sub_y == sub_y[0]): # fix !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth: # add depth
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if N < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): # fix (1, ...)
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count # fix poryadok
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # fix sortirovka po klycham
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) # fix list
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if self._min_samples_leaf is not None:
                    if np.sum(split) < self._min_samples_leaf or np.sum(~split) < self._min_samples_leaf:
                        continue

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": # fix uppercase
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # fix 
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth=depth+1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth=depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_index = node["feature_split"]
        feature_type = self._feature_types[feature_index]
        
        if feature_type == "real":
            if x[feature_index] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_index] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type in prediction")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **parameters):
        if 'feature_types' in parameters:
            self._feature_types = parameters['feature_types']
        if 'max_depth' in parameters:
            self._max_depth = parameters['max_depth']
        if 'min_samples_split' in parameters:
            self._min_samples_split = parameters['min_samples_split']
        if 'min_samples_leaf' in parameters:
            self._min_samples_leaf = parameters['min_samples_leaf']
            
        return self