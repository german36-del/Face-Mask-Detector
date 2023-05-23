########################
#
#      --- - FACEMASK DETECTOR - ---
#   
#   Classification functions
#       - Training by the pre-parsed data in preprocessing
#       - Classification of new data
#
####

import numpy as np
import os
import cv2
import pickle
import mediapipe as mp
import FMD_preprocessing as pp
import pandas as pd



from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
# For plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier



## Import data
def KNN_classi(X,y,data):
    
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Print the length and width of our testing data.
    print('Length of our Training data: ', len(X_train), '\nLength of our Testing data: ', len(X_test))
    best_params = {'weights': 'distance', 'n_neighbors': 2, 'metric': 'manhattan'}
# create new a knn model with best params
    b_knn = KNeighborsClassifier(**best_params)

#fit model to data
    b_knn.fit(X_train, y_train)

# make prediction on entire test data
    train_pred = b_knn.predict(X_train)

# make prediction on entire test data
    y_pred = b_knn.predict(X_test)

    print('Accuracy Train: %.3f' % accuracy_score(y_train, train_pred))
    print('Accuracy Test: %.3f' % accuracy_score(y_test, y_pred))
    print("\nClassification Report\n", classification_report(y_test, y_pred))
# Calculate Confusion Matrix for Best Param Model
    m = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9,9))
# Heatmap visualization of accuracy
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Greens_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    title = 'Accuracy Score Best Params: {0}'.format(accuracy_score(y_test, y_pred))
    plt.title(title,size=15)

    return b_knn
    
if(__name__=="__main__"):
    pickle_in = open("output/data.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("output/y.pickle", "rb")
    y = pickle.load(pickle_in)
    pickle_in = open("output/data.pickle", "rb")
    data = pickle.load(pickle_in)
    model=KNN_classi(X,y,data)
    
    with open("output/model.pickle", "wb") as out:
        pickle.dump(model,out)