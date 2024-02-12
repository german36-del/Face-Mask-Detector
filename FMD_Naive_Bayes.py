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
import sklearn
import sklearn.naive_bayes
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score

# For plotting
import matplotlib.pyplot as plt

## Import data
def Naive_bayes(X,y):
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # GaussianNB Building
    gaussianNB_model = sklearn.naive_bayes.GaussianNB()

# Training
    gaussianNB_model.fit(X_train, y_train)
    gnb_y_pred = gaussianNB_model.predict(X_test)

    accuracy = gaussianNB_model.score(X_test, y_test)
    print("Accuracy %f" % accuracy)
    metrics.accuracy_score(y_true=y_test, y_pred=gnb_y_pred)

    
    cm = confusion_matrix(y_test, gnb_y_pred)

    plt.figure(figsize=(9,9))
# Heatmap visualization of accuracy
    sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Reds_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_true=y_test, y_pred=gnb_y_pred))
    plt.title(all_sample_title,size=15)

    return gaussianNB_model
   
    


if(__name__=="__main__"):
    sns.set()
    pickle_in = open("output/data.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("output/y.pickle", "rb")
    y = pickle.load(pickle_in)
    print('Type X:', type(X))
    print('Type y:', type(y), end='\n\n')
    ##Hay un problema, necesita la imagen entera no un vector de características
    
    
    
    model=Naive_bayes(X,y)
    
    with open("output/model.pickle", "wb") as out:
        pickle.dump(model,out)

