########################
#
#      --- - FACEMASK DETECTOR - ---
#   
#   Classification functions
#       - Training by the pre-parsed data in preprocessing
#       - Classification of new data
#
####

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import seaborn as sns # for confusion matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # to plot image, graph

import pickle
import time # for computation time assessment


## Import data
def Decision_Tree(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
# Initialize Decision Trees model, No hyperparameter Tuning
    decision_trees = DecisionTreeClassifier()

# Use training data to fit Decision Trees model
    decision_trees.fit(X_train, y_train)

# Predict Train Data Labels
    predictions_set = decision_trees.predict(X_test)
# Calculate Confusion Matrix
    cm = confusion_matrix(y_test, predictions_set)

    plt.figure(figsize=(9,9))
# Heatmap visualization of accuracy
    sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    title = 'Accuracy Score, No Hyperparameter Tuning: {0}'.format(accuracy_score(y_test, predictions_set))
    plt.title(title,size=15)
    print('Decision Trees Precision: %.3f' % precision_score(y_test, predictions_set, average='micro'))
    print('Decision Trees Recall: %.3f' % recall_score(y_test, predictions_set, average='micro'))
    print('Decision Trees F1 Score: %.3f' % f1_score(y_test, predictions_set, average='micro'))
    print("\nClassification Report\n", classification_report(y_test, predictions_set))
    
    return decision_trees

if(__name__=="__main__"):
    pickle_in = open("output/data.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("output/y.pickle", "rb")
    y = pickle.load(pickle_in)

    model = Decision_Tree(X,y)
    
    with open("output/model.pickle", "wb") as out:
        pickle.dump(model,out)