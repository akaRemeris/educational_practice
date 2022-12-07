from decision_tree import *


class DecisionTreeClassifier(DecisionTree):

    def __init__(self, criterion, max_depth):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root_node = DTCNode(X, y)
        self.root_node.split(self.max_depth)

    def predict(self, X):
        self.root_node.predict(X)


