import math


class DTCNode(object):
    def __init__(self, X, y, current_depth=0, node_feature=None):
        self.X = X
        self.y = y
        self.entropy = self.get_entropy(y)
        self.left = None
        self.right = None
        self.node_feature = node_feature
        self.current_depth = current_depth
        try:
            self.probability = int(self.y.sum())/int(self.y.shape[0])
        except ZeroDivisionError:
            self.probability = 1

    @staticmethod
    def get_entropy(y):
        p = int(y.sum()) / int(y.shape[0])
        try:
            return - p * math.log2(p) - (1 - p) * math.log2(1-p)
        except ValueError as e:
            return 0

    def split(self, max_depth):
        ig_log = list()

        # search for best feature using entropy, putting in ig_log
        for feature in self.X:

            entropy_true = self.entropy
            entropy_false = self.entropy

            size_ratio_true = self.X.loc[self.X[feature] == 1].shape[0] / self.X.shape[0]
            if size_ratio_true != 0:
                entropy_true = self.get_entropy(self.y.loc[self.X[feature] == 1])

            size_ratio_false = self.X.loc[self.X[feature] == 0].shape[0] / self.X.shape[0]
            if size_ratio_false != 0:
                entropy_false = self.get_entropy(self.y.loc[self.X[feature] == 0])

            ig = self.entropy - size_ratio_true * entropy_true - size_ratio_false * entropy_false
            ig_log.append(ig)

        # split by best feature
        split_feature_index = ig_log.index(max(ig_log))
        self.node_feature = self.X.columns[split_feature_index]

        # get split data where true
        true_split = self.X.iloc[:, split_feature_index] == 1
        new_true_data = self.X.loc[true_split, :].drop(self.X.columns[split_feature_index], axis=1)

        # get split data where false
        false_split = self.X.iloc[:, split_feature_index] == 0
        new_false_data = self.X.loc[false_split, :].drop(self.X.columns[split_feature_index], axis=1)

        # proceed splitting
        if self.current_depth < max_depth:

            if new_true_data.shape[0] != 0:
                self.left = DTCNode(new_true_data, self.y.loc[true_split], self.current_depth + 1, self.node_feature)
            if new_true_data.shape[1] > 0:
                self.left.split(max_depth)

            if new_false_data.shape[0] != 0:
                self.right = DTCNode(new_false_data, self.y.loc[false_split], self.current_depth + 1, self.node_feature)
            if new_false_data.shape[1] > 0:
                self.right.split(max_depth)

    def predict(self, X):
        if X[self.node_feature][0]:
            if self.left is None:
                print(self.probability)
            else:
                self.left.predict(X)
        else:
            if self.right is None:
                print(self.probability)
            else:
                self.right.predict(X)
