import mysklearn.myutils as myutils
import math
import numpy as np
import copy
import random

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       # TODO: seed your random number generator
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
       np.random.seed(random_state)
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        index_shuffle = list(range(len(X)))
        shuffled_X = []
        shuffled_y = []
        random.shuffle(index_shuffle)
        for i in index_shuffle:
            shuffled_X.append(X[i])
            shuffled_y.append(y[i])

        X = copy.deepcopy(shuffled_X)
        y = copy.deepcopy(shuffled_y)

    num_instances = len(X) 
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) # ceil(8 * 0.33)
    split_index = num_instances - test_size # 8 - 2 = 6

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]
    

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    
    X_test_folds = []
    X_train_folds = []
    # add empty buckets to list 
    for i in range(n_splits):
        X_test_folds.append([])
        X_train_folds.append([])
    # append 
    for i in range(len(X)):
        X_test_folds[i % n_splits].append(i)
    for i in range(n_splits):
        for j in range(len(X)):
            if j not in X_test_folds[i]:
                X_train_folds[i].append(j)
       
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # create dictionary where key is class label and value is empty list
    y_dict = myutils.create_dictionary(y)
    y_indices = copy.deepcopy(y_dict)
    for key in y_indices:
        y_indices[key] = []
    # put indexes list according to their class in y
    for i in range(len(X)):
        y_indices[y[i]].append(i)

    # for each value in dictionary, put indices into buckets 
    X_test_folds = []
    X_train_folds = []

    # add empty buckets to list 
    for i in range(n_splits):
        X_test_folds.append([])
        X_train_folds.append([])

    # append 
    for item in y_indices:
        i = 0
        while i < len(y_indices[item]):
            for j in range(len(X_test_folds)):
                if i == len(y_indices[item]):
                    break
                X_test_folds[j].append(y_indices[item][i])
                i += 1
    for i in range(n_splits):
        for j in range(len(X)):
            if j not in X_test_folds[i]:
                X_train_folds[i].append(j)

def stratified_test_remainder(X, y, n_splits=3, shuffle=True):
    # test set and remainder set are a list of indices 
    test_set = []
    remainder_set = []
    # create dictionary where key is class label and value is empty list
    y_dict = myutils.create_dictionary(y)
    y_indices = copy.deepcopy(y_dict)
    for key in y_indices:
        y_indices[key] = []
    # put indices in list according to their class in y
    for i in range(len(X)):
        y_indices[y[i]].append(i)

    # shuffle the indices
    for item in y_indices:
        random.shuffle(y_indices[item])
    
    # put indices in sets with 1/3 in test and 2/3 in remainder 
    for item in y_indices:
        for i in range(int(len(y_indices[item]) / n_splits)):
            test_set.append(y_indices[item][i])
        for i in range(int(len(y_indices[item]) / n_splits), len(y_indices[item])):
            remainder_set.append(y_indices[item][i])
      
    
    return test_set, remainder_set

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    # make dictionary with labels as keys and indices as values for easy lookup
    index_dict = {}
    for i in range(len(labels)):
        index_dict[labels[i]] = i

    for label in labels:
        row = []
        for item in labels:
            row.append(0)
        for i in range(len(y_true)):
            if y_true[i] == label:
                if y_pred[i] == None:
                    pass
                else:
                    row[index_dict[y_pred[i]]] += 1
        matrix.append(row)

    return matrix