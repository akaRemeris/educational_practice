import pandas as pd
import numpy as np
from decision_tree_classifier import *


df = pd.DataFrame({'X1': [1, 1, 1, 0, 0, 0, 0, 1],
                   'X2': [0, 1, 1, 0, 0, 1, 0, 1],
                   # 'X3': [0, 0, 1, 0, 0, 1, 0, 0],
                   'Y':  [1, 1, 1, 1, 0, 0, 0, 0]})
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf.fit(df.iloc[:, :-1], df['Y'])
sample = pd.DataFrame({'X1': [1], 'X2': [1]})
clf.predict(sample)
