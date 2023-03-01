import importlib
import numpy as np

import utils as u
importlib.reload(u)


train, test = u.load_data()

# information_gain or random
measure = "random"
attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int)

tree = u.learn_decision_tree(train, attributes, None, None, None, measure)

print(u.accuracy(tree, test))
