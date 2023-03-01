import importlib
import numpy as np

import assignment_6 as u
importlib.reload(u)


train, test = u.load_data()

# information_gain or random
measure = "random"
attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int)

tree = u.learn_decision_tree(train, attributes, None, None, None, measure)

accuracies = []
for i in range(10) : 
    tree = u.learn_decision_tree(train, attributes, None, None, None, measure)
    accuracies.append(u.accuracy(tree, train)[0])

print(accuracies)