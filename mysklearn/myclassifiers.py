import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
import copy
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        x_avg = 0
        y_avg = 0
        count = 0
        for row in X_train:
            for X in row:
                x_avg += X
                count += 1

        x_avg = x_avg / count
        y_avg = sum(y_train) / len(y_train)
        numerator = 0
        denominator = 0
        m = 0

        for i in range(0, len(X_train)):
            for j in range(0, len(X_train[0])):
                numerator += (X_train[i][j] - x_avg) * (y_train[i] - y_avg)
        
        for i in range(0, len(X_train)):
            for j in range(0, len(X_train[0])):
                denominator += (X_train[i][j] - x_avg) ** 2
        
        m = numerator / denominator 
        b = y_avg - m * x_avg

        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        y = 0

        for row in X_test:
            for X in row:
                y = self.slope * X + self.intercept
                y_predicted.append(y)
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        
        neighbor_indices = []
        

        for test_row in X_test:
            row_distances = []
            row_indicies = []
            for i in range(len(self.X_train)):
                d = myutils.compute_euclidean_distance(self.X_train[i], test_row)
                row_distances.append(d)
                row_indicies.append(i)
            sorted_distances, sorted_indices = (list(tup) for tup in zip(*sorted(zip(row_distances, row_indicies))))
            
            distances.append(sorted_distances[:self.n_neighbors])
            neighbor_indices.append(sorted_indices[:self.n_neighbors])
        
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, neighbor_indices = self.kneighbors(X_test)
        
        for i in range(len(neighbor_indices)):
            labels = [self.y_train[i] for i in neighbor_indices[i]]
            winner = myutils.most_common(labels)
            y_predicted.append(winner)
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # dictionary where the key is the class label and value is the probability
        self.X_train = X_train
        self.y_train = y_train
        self.priors = {}
        # dictionary where key is the class label and value is the number of occurences
        class_dict = myutils.create_dictionary(y_train)
        total_instances = len(y_train)

        # copy class_dict into priors
        self.priors = copy.deepcopy(class_dict)
        # divide value by total instances to get percent 
        for item in self.priors:
            self.priors[item] = self.priors[item]/total_instances
        #print(self.priors)
        # value of nested dictionary is probability of class label, parallel to labels
        labels = self.priors.keys()
        self.posteriors = {}

        # create a dictionary of attributes and how often they appear 
        # attributes = {att0 : {1 : 5, 2 : 10}, att1 : {3 : 5, 2 : 6, 1: 4}}
        attribute_totals = {}
        for i in range(len(X_train[0])):
            name = "att{}".format(i)
            attribute_totals[name] = {}
            for j in range(len(X_train)):
                if X_train[j][i] in attribute_totals[name]:
                    attribute_totals[name][X_train[j][i]] += 1
                else:
                    attribute_totals[name][X_train[j][i]] = 1

        # create a dictionary with posterior probabilities
        # self.posteriors = {att0 : {1 : {"yes" : 0, "no" : 0}, 2 : {"yes" : 0, "no" : 0}}
        self.posteriors = copy.deepcopy(attribute_totals)
        for attribute in self.posteriors:
            for item in self.posteriors[attribute]:
                self.posteriors[attribute][item] = {}
                for label in labels:
                    self.posteriors[attribute][item][label] = 0
        
        
        for i in range(len(X_train[0])): # iterate through attributes
            for item in self.posteriors["att{}".format(i)]: # iterate through dictionary at attribute
                for row in range(len(X_train)): # iterate down rows
                    if X_train[row][i] == item:
                        self.posteriors["att{}".format(i)][item][y_train[row]] += 1
                for label in labels:
                    self.posteriors["att{}".format(i)][item][label] = self.posteriors["att{}".format(i)][item][label] / (self.priors[label] * total_instances)

        #print(self.posteriors)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        # iterate through posteriors and multiplying by the value in labels 
        for X in X_test:
            # dictionary holding probability of each label 
            labels = copy.deepcopy(self.priors)
            for attribute in self.posteriors:
                # attributes are labeled 'att0', 'att1', 'att2', etc. 
                # removing 'att' will give the index of the attribute in X_test
                name = int(attribute.replace("att", ""))
                for item in self.posteriors[attribute]:
                    if X[name] == item:
                        for label in self.posteriors[attribute][item]:
                            labels[label] = labels[label] * self.posteriors[attribute][item][label]
            # gets key with highest percent 
            decision = max(labels, key=labels.get)
            y_predicted.append(decision)

        return y_predicted
    

class MyZeroClassifier():
    def __init__(self):
        """Initializer for MyZeroClassifier.
        """ 
        self.y_train = None
        self.X_train = None

    def fit(self, X_train, y_train):
        self.y_train = y_train
        self.X_train = X_train

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        count = myutils.create_dictionary(self.y_train)
        Keymax = max(count, key=count.get)
        # get key with max value
        max_val = count[Keymax]
        # if there are multiple keys with that value, pick a random one 
        max_lst = []
        for item in count:
            if count[item] == max_val:
                max_lst.append(item)
        y_predicted = random.choice(max_lst)

        return y_predicted

class MyRandomClassifier():
    def __init__(self):
        """Initializer for MyRandomClassifier.
        """ 
        self.y_train = None
        self.X_train = None

    def fit(self, X_train, y_train):
        self.y_train = y_train
        self.X_train = X_train

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = random.choice(self.y_train)

        return y_predicted

class MyDecisionTreeClassifier():
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None
        self.header = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # fit() accepts X_train and y_train
        self.X_train = X_train
        self.y_train = y_train
        # compute the attribute domains dictionary
        domain_dict = {}
        for col in range(len(X_train[0])):
            name = "att{}".format(col)
            domain_dict[name] = []
            for row in range(len(X_train)):
                if X_train[row][col] not in domain_dict[name]:
                    domain_dict[name].append(X_train[row][col])

        # compute a "header" ["att0", "att1", ...]
        self.header = list(domain_dict.keys())
        # my advice is to stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # initial call to tdidt current instances is the whole table (train)
        available_attributes = self.header.copy() # python is pass object reference
        self.tree = myutils.tdidt(train, available_attributes, domain_dict, self.header)

        # print(self.tree)
       
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for X in X_test:
            y_predicted.append(myutils.tdidt_predict(self.header, self.tree, X))
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForest():
    def __init__(self):
        """Initializer for MyRandomForest.
        """ 
        self.y_train = None
        self.X_train = None
        self.forest = []
        self.full_forest = []
        self.table = []
        self.header = []

    def fit(self, X_train, y_train, N, M, F, header):
        self.y_train = y_train
        self.X_train = X_train
        self.table = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.header = header

        # list of performance stats, parallel with full_forest
        performance = []
        for i in range(N):
            DecisionTree = MyDecisionTreeClassifier()
            # get subset of columns
            att_indexes = list(range(len(header)))
            subset = myutils.compute_random_subset(att_indexes, F)
            subset.append(len(header))
            X_subset = []
            
            for i in range(len(self.table)):
                new_row = []
                for j in range(len(self.table[0])):
                    for index in subset:
                        if j == index:
                            new_row.append(self.table[i][j])
                X_subset.append(new_row)

            
            # get 67% of remainder set for sample
            sample = myutils.compute_bootstrapped_sample(X_subset)
            X_sample, y_sample = myutils.separate_data_from_class(sample)
            
           
            # get remaining 33% for validation
            validation = []
            for item in X_subset:
                if item not in X_sample:
                    validation.append(item)
            #print("validation", validation)
            X_test, y_test = myutils.separate_data_from_class(validation)
            DecisionTree.fit(X_sample, y_sample)
            self.full_forest.append(DecisionTree)

            predictions = DecisionTree.predict(X_test)
            percent_correct = myutils.get_percent_correct(predictions, y_test)
            performance.append(percent_correct)
        
        # get M best trees in full_forest and append to forest
        for i in range(M):
            best_index = performance.index(max(performance))
            self.forest.append(self.full_forest[best_index])
            del performance[best_index]
            del self.full_forest[best_index]


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for X in X_test:
            predictions = []
            for decision_tree in self.forest:
                prediction = decision_tree.predict([X])
                predictions.append(prediction[0])
            majority_vote = myutils.most_frequent(predictions)
            y_predicted.append(majority_vote)
        return y_predicted

