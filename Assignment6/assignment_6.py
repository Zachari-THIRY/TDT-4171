import numpy as np
from pathlib import Path
from typing import Tuple



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)
    def print(self):
        print("Node : ")
        print(f"\tFrom parent attribute {self.parent.attribute}")
        print(f"\tSplits on attribute : {self.attribute}")
        print(f"\tReturns value: {self.value}")



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)
        
    Return
    ------
    The most common value label within `examples`.
    """
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    assert measure in ["random", "information_gain"], "Parameter `measure` should be either 'random' or 'information_gain"

    # Implementing the measure = "random"
    if measure == "random" : return np.random.choice(range(len(attributes)-1))

    # Implementing the measure = "information_gain"
    H = entropy(examples)
    attributes_gains = [H-remainder(attribute, examples) for attribute in attributes]

    return np.argmax(attributes_gains)



def entropy(examples:np.ndarray) :
    """Computes the entropy of the given array
    
    Parameters
    ----------
        examples : np.ndarray
            The examples array, containing the label in the last column
    Return
    ------
        float : the entropy value for the given set of examples
    """
    labels = examples[:, -1]
    L = np.shape(examples)[0]
    P = []
    for label in np.unique(labels):
        P.append((np.count_nonzero(labels == label))/L)
    
    H = 0
    for p in P:
        H += p * np.log2(1/p)
    return H

def boolean_entropy(q):
    if q == 0 or  q== 1 : return 0
    return -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(attribute:int, examples):
    """ Computes the remainder of subsetting examples according to the attribute
    """
    n = np.count_nonzero(examples[:,-1] == 1)
    p = np.count_nonzero(examples[:,-1] == 2)

    R = 0
    try :
        for value in [2,1]:
            subset = examples[np.where(examples[:,attribute]==value)]
            subset_labels = subset[:,-1]
            nk = np.count_nonzero(subset_labels == 1)
            pk = np.count_nonzero(subset_labels == 2)

            R += (pk + nk)/(p+n) * boolean_entropy((pk)/(pk+nk))
    except:
        print(examples, attribute)
        raise ValueError
    return R


def entropy(examples:np.ndarray):
    """Computes the entropy of the given array
    
    Parameters
    ----------
        examples : np.ndarray
            The examples array, containing the label in the last column
    Return
    ------
        float : the entropy value for the given set of examples
    """

    labels = examples[:, -1]
    L = np.shape(examples)[0]
    P = []
    for label in np.unique(labels):
        P.append((np.count_nonzero(labels == label))/L)
    
    H = 0
    for p in P:
        H += p * np.log2(1/p)
    return H

def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678
    # If examples is empty, return the plurality of the parent examples
    if np.shape(examples)==0:
        node.value = plurality_value(parent_examples)
    # If all examples have the same classification
    elif np.shape(np.unique(examples[:,-1]))[0] == 1 :
        node.value = np.unique(examples[:,-1])
    # If attributes are empty
    elif np.shape(attributes)[0] == 0 :
        node.value = plurality_value(examples)
    # Otherwise, create a reccurvie node
    else :
        A_index = importance(attributes, examples, measure)
        A = attributes[A_index]
        attributes = np.delete(attributes,A_index)

        node.attribute = A
        values_of_A = np.unique(examples[:,A])
        for v in values_of_A:
            exs = examples[np.where(examples[:,A]==v)]
            node.children[v] = learn_decision_tree(exs, attributes, examples, node, v, measure)
    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test


if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "information_gain"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")