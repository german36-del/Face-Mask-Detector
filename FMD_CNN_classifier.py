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
import matplotlib.pyplot as plt

import tensorflow
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import os

## Import data
def CNN(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cnn_model = tensorflow.keras.models.Sequential()
    # Start of Convolution Layers & Maxpooling
    cnn_model.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.MaxPool2D())

    cnn_model.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.MaxPool2D())

    cnn_model.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.MaxPool2D())

# Start of Neural Nets
    cnn_model.add(tensorflow.keras.layers.Flatten())

    cnn_model.add(tensorflow.keras.layers.Dense(512, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Dropout(0.3))
    cnn_model.add(tensorflow.keras.layers.Dense(512, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Dropout(0.3))
    cnn_model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    cnn_model.add(tensorflow.keras.layers.Dense(3, activation='softmax'))
    #Enable this to see the summary of the model
    #cnn_model.summary()
    cnn_model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])
    epochs = 10
    cnn_model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)
    cnn_model.evaluate(X_test, y_test)
    y_pred = np.argmax(cnn_model.predict(X_test), axis=-1)
    accuracy = cnn_model.score(X_test, y_test)
    print("Accuracy %f" % accuracy)
    metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
#To see the heatmap, put a breakpoint here
    m = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9,9))
# Heatmap visualization of accuracy
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Greens_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    title = 'Accuracy Score Best Params: {0}'.format(accuracy_score(y_test, y_pred))
    plt.title(title,size=15)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    
    return cnn_model
if(__name__=="__main__"):
    pickle_in = open("output/data.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("output/y.pickle", "rb")
    y = pickle.load(pickle_in)

    model = CNN(X,y)
    
    with open("output/model.pickle", "wb") as out:
        pickle.dump(model,out)