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

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
# For plotting
import matplotlib.pyplot as plt

## Import data
def SVM_train(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model_100 = svm.SVC()
    model_100.fit(X_train, y_train)

    y_pred = model_100.predict(X_test)
    accuracy = model_100.score(X_test, y_test)
    print("Accuracy %f" % accuracy)
    metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9,9))
# Heatmap visualization of accuracy
    sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Reds_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_true=y_test, y_pred=y_pred))
    plt.title(all_sample_title,size=15)

    return model_100


if __name__== "__main__":
    pickle_in = open("output/data.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("output/y.pickle", "rb")
    y = pickle.load(pickle_in)

    model = SVM_train(X,y)
    with open("output/model.pickle", "wb") as out:
        pickle.dump(model,out)