import numpy as np
import scipy.stats as stats 
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForest
"""
# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    assert False == True # TODO: copy your solution from PA4 here

def test_simple_linear_regressor_predict():
    assert False == True # TODO: copy your solution from PA4 here

def test_kneighbors_classifier_kneighbors():
    assert False == True # TODO: copy your solution from PA4 here

def test_kneighbors_classifier_predict():
    assert False == True # TODO: copy your solution from PA4 here

def test_naive_bayes_classifier_fit():
    NaiveBayes = MyNaiveBayesClassifier()

    # 8 element dataset
    train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

    NaiveBayes.fit(train, train_labels)

    desk_priors = {"no": 4/8, "yes": 4/8}
    desk_posteriors = {"att0": {3: {"no": 1/4, "yes": 0/4}, 6: {"no": 0/4, "yes": 1/4}, 
                                4: {"no": 2/4, "yes": 0/4}, 1: {"no": 0/4, "yes": 2/4}, 
                                2: {"no": 1/4, "yes": 0/4}, 0: {"no": 0/4, "yes": 1/4}}, 
                        "att1": {2: {"no": 1/4, "yes": 1/4}, 6: {"no": 0/4, "yes": 2/4}, 
                                1: {"no": 1/4, "yes": 0/4}, 4: {"no": 1/4, "yes": 0/4}, 
                                0: {"no": 1/4, "yes": 0/4}, 3: {"no": 0/4, "yes": 1/4}}}
    
    assert NaiveBayes.priors == desk_priors
    assert NaiveBayes.posteriors == desk_posteriors  
    print("Passed Naive Bayes fit for 8 element training set.")  

    # RQ5 training set 
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_X = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
    ]

    iphone_y = ["no", "no", "yes", "yes", 
                "yes", "no", "yes", "no", 
                "yes", "yes", "yes", "yes", 
                "yes", "no", "yes"]
    
    NaiveBayes.fit(iphone_X, iphone_y)

    # fit function adds to dictionary in order of occurrence 
    desk_priors = {"no": 5/15, "yes": 10/15}
    desk_posteriors =  {"att0": {1: {"no": 3/5, "yes": 2/10}, 2: {"no": 2/5, "yes": 8/10}}, 
                        "att1": {3: {"no": 2/5, "yes": 3/10}, 2: {"no": 2/5, "yes": 4/10}, 1: {"no": 1/5, "yes": 3/10}},
                        "att2": {"fair": {"no": 2/5, "yes": 7/10}, "excellent": {"no": 3/5, "yes": 3/10}}}
    
    assert NaiveBayes.priors == desk_priors
    assert NaiveBayes.posteriors == desk_posteriors
    print("Passed Naive Bayes fit for RQ5 training set.")

    Bramer_col_names = ["day", "season", "wind", "rain", "class"]
    Bramer_X = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"], 
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    Bramer_y = ["on time", "on time", "on time", "late", "on time", 
                "very late", "on time", "on time", "very late", "on time",
                 "cancelled", "on time", "late", "on time", "very late", 
                 "on time", "on time", "on time", "on time", "on time"]

    NaiveBayes.fit(Bramer_X, Bramer_y)  

    desk_priors = {"on time": 14/20, "late": 2/20, "very late": 3/20, "cancelled": 1/20}
    desk_posteriors =  {"att0": {"weekday": {"on time": 9/14, "late": 1/2, "very late": 3/3, "cancelled": 0/1}, 
                                "saturday": {"on time": 2/14, "late": 1/2, "very late": 0/3, "cancelled": 1/1}, 
                                "holiday": {"on time": 2/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1}, 
                                "sunday": {"on time": 1/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1}}, 
                        "att1": {"spring": {"on time": 4/14, "late": 0/2, "very late": 0/3, "cancelled": 1/1}, 
                                "winter": {"on time": 2/14, "late": 2/2, "very late": 2/3, "cancelled": 0/1}, 
                                "summer": {"on time": 6/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1}, 
                                "autumn": {"on time": 2/14, "late": 0/2, "very late": 1/3, "cancelled": 0/1}},
                        "att2": {"none": {"on time": 5/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1}, 
                                "high": {"on time": 4/14, "late": 1/2, "very late": 1/3, "cancelled": 1/1}, 
                                "normal": {"on time": 5/14, "late": 1/2, "very late": 2/3, "cancelled": 0/1}},
                        "att3": {"none": {"on time": 5/14, "late": 1/2, "very late": 1/3, "cancelled": 0/1}, 
                                "slight": {"on time": 8/14, "late": 0/2, "very late": 0/3, "cancelled": 0/1}, 
                                "heavy": {"on time": 1/14, "late": 1/2, "very late": 2/3, "cancelled": 1/1}}}
    
    assert NaiveBayes.priors == desk_priors
    assert NaiveBayes.posteriors == desk_posteriors
    print("Passed Naive Bayes fit for Bramer.")

def test_naive_bayes_classifier_predict():
    NaiveBayes = MyNaiveBayesClassifier()

    # 8 element dataset
    train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    train_test = [[2, 3]]
    desk_predict = ["no"]
    NaiveBayes.fit(train, train_labels)
    prediction = NaiveBayes.predict(train_test)
    assert prediction == desk_predict
    print("Passed Naive Bayes predict for 8 element training set.")

    # RQ5 training set 
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_X = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
    ]

    iphone_y = ["no", "no", "yes", "yes", 
                "yes", "no", "yes", "no", 
                "yes", "yes", "yes", "yes", 
                "yes", "no", "yes"]
    
    iphone_test = [[2, 2, "fair"]]
    NaiveBayes.fit(iphone_X, iphone_y)
    prediction = NaiveBayes.predict(iphone_test)
    desk_predict = ["yes"]
    assert prediction == desk_predict
    print("Passed Naive Bayes predict for RQ5 training set.")

    # Bramer training set 
    Bramer_X = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"], 
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    Bramer_y = ["on time", "on time", "on time", "late", "on time", 
                "very late", "on time", "on time", "very late", "on time",
                 "cancelled", "on time", "late", "on time", "very late", 
                 "on time", "on time", "on time", "on time", "on time"]
    Bramer_test = [["weekday", "winter", "high", "heavy"]]
    NaiveBayes.fit(Bramer_X, Bramer_y)
    prediction = NaiveBayes.predict(Bramer_test)
    desk_predict = ["very late"]
"""
def test_decision_tree_classifier_fit():
    X_train = [

        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y_train = ["False", "False", "True", "True", "True", "False", "True", "False",

            "True", "True", "True", "True", "True", "False"]

    interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

    [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "no", "True"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    DecisionTree = MyDecisionTreeClassifier()
    #DecisionTree.fit(X_train, y_train)
    #assert DecisionTree.tree == interview_tree
    #print("Passed fit test for interview tree.")
    #DecisionTree.print_decision_rules()
    
    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project"]
    degrees_table = [
            ["A", "B", "A", "B", "B"],
            ["A", "B", "B", "B", "A"],
            ["A", "A", "A", "B", "B"],
            ["B", "A", "A", "B", "B"],
            ["A", "A", "B", "B", "A"],
            ["B", "A", "A", "B", "B"],
            ["A", "B", "B", "B", "B"],
            ["A", "B", "B", "B", "B"],
            ["A", "A", "A", "A", "A"],
            ["B", "A", "A", "B", "B"],
            ["B", "A", "A", "B", "B"],
            ["A", "B", "B", "A", "B"],
            ["B", "B", "B", "B", "A"],
            ["A", "A", "B", "A", "B"],
            ["B", "B", "B", "B", "A"],
            ["A", "A", "B", "B", "B"],
            ["B", "B", "B", "B", "B"],
            ["A", "A", "B", "A", "A"],
            ["B", "B", "B", "A", "A"],
            ["B", "B", "A", "A", "B"],
            ["B", "B", "B", "B", "A"],
            ["B", "A", "B", "A", "B"],
            ["A", "B", "B", "B", "A"],
            ["A", "B", "A", "B", "B"],
            ["B", "A", "B", "B", "B"],
            ["A", "B", "B", "B", "B"],
        ]
    
    degrees_classes = ["SECOND", "FIRST", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", 
                        "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", 
                        "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", 
                        "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", ]

    degrees_tree = \
        ['Attribute', 'att0',
        ['Value', 'A',
            ['Attribute', 'att4',
            ['Value', 'A', ['Leaf', 'FIRST', 5, 14]],
            ['Value', 'B',
                ['Attribute', 'att3',
                ['Value', 'A',
                    ['Attribute', 'att1',
                    ['Value', 'A', ['Leaf', 'FIRST', 1, 2]],
                    ['Value', 'B', ['Leaf', 'SECOND', 1, 2]]]],
            ['Value', 'B', ['Leaf', 'SECOND', 7, 9]]]]]],
        ['Value', 'B', ['Leaf', 'SECOND', 12, 26]]]

    X_test_degrees = [["B", "B", "B", "B", "B"], 
                    ["A", "A", "A", "A", "A"], 
                    ["A", "A", "A", "A", "B"]]

    y_test_degrees = ['SECOND', 'FIRST', 'FIRST']

    DecisionTree.fit(degrees_table, degrees_classes)
    assert DecisionTree.tree == degrees_tree
    print("Passed fit test for interview tree.")



def test_decision_tree_classifier_predict():
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    test_1 = [["Junior", "Java", "yes", "no"]]
    actual_1 = ["True"]
    test_2 = [["Junior", "Java", "yes", "yes"]]
    actual_2 = ["False"]

    DecisionTree = MyDecisionTreeClassifier()
    DecisionTree.fit(X_train, y_train)
    predict_1 = DecisionTree.predict(test_1)
    predict_2 = DecisionTree.predict(test_2)
    assert predict_1 == actual_1
    assert predict_2 == actual_2
    print("Passed predict for interview set.")

    
    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project"]
    degrees_table = [
            ["A", "B", "A", "B", "B"],
            ["A", "B", "B", "B", "A"],
            ["A", "A", "A", "B", "B"],
            ["B", "A", "A", "B", "B"],
            ["A", "A", "B", "B", "A"],
            ["B", "A", "A", "B", "B"],
            ["A", "B", "B", "B", "B"],
            ["A", "B", "B", "B", "B"],
            ["A", "A", "A", "A", "A"],
            ["B", "A", "A", "B", "B"],
            ["B", "A", "A", "B", "B"],
            ["A", "B", "B", "A", "B"],
            ["B", "B", "B", "B", "A"],
            ["A", "A", "B", "A", "B"],
            ["B", "B", "B", "B", "A"],
            ["A", "A", "B", "B", "B"],
            ["B", "B", "B", "B", "B"],
            ["A", "A", "B", "A", "A"],
            ["B", "B", "B", "A", "A"],
            ["B", "B", "A", "A", "B"],
            ["B", "B", "B", "B", "A"],
            ["B", "A", "B", "A", "B"],
            ["A", "B", "B", "B", "A"],
            ["A", "B", "A", "B", "B"],
            ["B", "A", "B", "B", "B"],
            ["A", "B", "B", "B", "B"],
        ]
    
    degrees_classes = ["SECOND", "FIRST", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", 
                        "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", 
                        "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", 
                        "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", ]

def test_random_forest():
    # pasted from DecisionTreeFun
    header = ["level", "lang", "tweets", "phd"]
    attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
        "lang": ["R", "Python", "Java"],
        "tweets": ["yes", "no"], 
        "phd": ["yes", "no"]}
    X = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    

    test_set_indices, remainder_set_indices = myevaluation.stratified_test_remainder(X, y) 
    X_train = []
    y_train = []
    for item in remainder_set_indices:
        X_train.append(X[item])  
        y_train.append(y[item]) 
    X_test = []
    y_test = []
    for item in test_set_indices:
        X_test.append(X[item])  
        y_test.append(y[item]) 

    RandomForest = MyRandomForest() 
    RandomForest.fit(X_train, y_train, 10, 5, 2, header)
    
    prediction_lst = RandomForest.predict(X_test)
    print("prediction", prediction_lst)
    

def main():
    #test_decision_tree_classifier_fit()
    #test_decision_tree_classifier_predict()
    test_random_forest()
main()
