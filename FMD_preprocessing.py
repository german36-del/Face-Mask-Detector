########################
#
#      --- - FACEMASK DETECTOR - ---
#   
#   Preprocessing functions
#       - Facial Landmarks with Mediapie
#       - Basic features
#
####

import numpy as np
import os
import cv2
import pickle
import mediapipe as mp

def create_data(CATEGORIES, DIRECTORY, N):
    '''
    Parse the data
    '''
    X = []
    y = []
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        class_num_label = CATEGORIES.index(category)
        n = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                #img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                X.append(img_array)
                y.append(class_num_label)
                n += 1
                if(n >= N):
                    break
            except Exception as e:
                pass
    return X,y


def getFeatures(img):
    '''
    Extracts the features of an image
    '''
    features = []
    ## Características
    features = features + avg(img)
    features = features + avgUpperHalf(img)
    features = features + avgLowerHalf(img)
    features = features + var(img)
    features = features + facialLandmarksFeatures(img)
    ##
    #features=np.concatenate((features,Harris(img,2,3,0.04,0.01)))
    return features




def avg(img):
    '''
    Mean value of the channels of the img  
    '''
    channels = cv2.split(img)
    features = []
    for c in channels:
        aux=np.array(c).flatten()
        features.append(np.mean(aux))
    return features

def avgUpperHalf(img):
    '''
    Mean value of the channels of the upper half img
    '''
    channels = cv2.split(img)
    features = []
    for c in channels:
        aux=np.array(c).flatten()
        features.append(np.mean(aux [: len(aux)//2]))
    return features

def avgLowerHalf(img):
    '''
    Mean value of the channels of the bottom half img
    '''
    channels = cv2.split(img)
    features = []
    for c in channels:
        aux=np.array(c).flatten()
        features.append(np.mean(aux [len(aux)//2 :]))
    return features

def var(img):
    '''
    Variance of the img
    '''
    return ([np.var(img)])

def Harris(img,blockSize,ksize,k,select):
    '''
    Esquinas de harris.
    @param: 
     blockSize: tamaño del vecindario
     ksize: apertura de la derivada de Sobel usada
     k: parámetro libre del detector de Harris
     select: porcentaje por encima del cuál se consideran puntos de interés 
    
    '''
    CH=[]
    Corners=[]
    # Prueba de parámetros que mejor resultados dan para Harris
    CH=cv2.cornerHarris(img,blockSize,ksize,k)
    flat= np.array(CH).flatten()

    for i in range(len(flat)):
        if flat[i]>select*CH.max():
            Corners.append(i)

    return(Corners)  # Devuelve un vector que ocupa un puesto en el vector resultado


def facialLandmarksFeatures(image, verbose=False):
    '''
    Get the facial landmarks' features
    '''
    mesh = mp.solutions.face_mesh
    face_mesh = mesh.FaceMesh()
    height, width, _ = image.shape

    if verbose:
        print("Height, width", height, width)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Funcion que da los facial_landmarks

    image_l = face_mesh.process(rgb_image)
    list=[]
    
    for face in image_l.multi_face_landmarks:
        for landmark in face.landmark:
            suma_b=0
            suma_g=0
            suma_r=0
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            list.append(x)
            list.append(y)
            for s in range(x-1,x+1):
                for t in range(y-1,y+1):

                    suma_b=suma_b+image.item(t,s,0)
                    suma_g=suma_g+image.item(t,s,1)
                    suma_r=suma_r+image.item(t,s,2)


            suma_b=suma_b/9
            suma_g=suma_g/9
            suma_r=suma_r/9     
            list.append(suma_b)
            list.append(suma_g)
            list.append(suma_r)

    return list

def saveData(X,y,data, path="output"):
    '''
    Saves the data to be used by the classifiers
    '''
    pickle_out = open(path+"/X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(path+"/y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_out = open(path+"/data.pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

if __name__ == "__main__":
    DIRECTORY = "C:\mixto" # Windows/PC
    CATEGORIES = ['IncorrectlyWorn', 'with_mask','without_mask']  
    print('Creating data structure')
    Xn, yn = create_data(CATEGORIES,DIRECTORY,2000)
    NBATCH = 487
    features = []
    X = []
    y = []
    print('Getting features: ')
    error = 0
    error_type = [0,0,0]
    types = [0,0,0]
    for i in range(len(Xn)):
        if types[yn[i]] < NBATCH:
            try:
                f = getFeatures(Xn[i])
                features.append(f)
                X.append(Xn[i])
                y.append(yn[i])
                types[yn[i]]+=1
            except:
                error_type[yn[i - error]]+=1
                error +=1
            
            #print("Error")
    print("Errors: ",error_type)
    print("Types: ", types)
    print("Features length: ", len(features))
    print("X length: ", len(X))
    print("y length: ", len(y))
    print('DONE!!')
    saveData(X,y,features)