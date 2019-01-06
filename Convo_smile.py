import numpy as np
import cv2
import os
from shutil import copy
from keras.preprocessing import image
import dlib
from imutils import face_utils
import imutils
from sklearn import svm
from sklearn.metrics import  classification_report,confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import  Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import optimizers
import time
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import regularizers
import pandas as pd

global basedir, image_paths, target_size,images_dir,labels_file
images_dir = 'D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces' #clean dataset
def format_data():
    features = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open('D:\\Documents\\Assignment1\\Lab21\\venv\\list.csv', 'r')
    lines = labels_file.readlines()
    all_labels = {line.split(',')[0]: int(line.split(',')[3]) for line in lines[2:]}# access smile label for each image
    labels={}
    i=0
    for img_path in image_paths:
        '''
        this commment out code was used to run greyscaled images but first have to run greying function
        '''
        # img= cv2.imread(img_path)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = gray/255
        # gray = gray.ravel()
        # features.append(gray) # list of all images
        file_name = img_path.split('.')[0].split('\\')[-1] # splits images into its corresponding classes by puting them in seperate folders
        if(all_labels[file_name] == 1):
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train\\smile')
        else:
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train\\No_smile')

def mlp(X,Y): # takes the validation and train generators as input argumnets
    #network created
    model = Sequential() # initalise model
    model.add(Conv2D(64,kernel_size= 3,activation='relu',strides=(2,2),input_shape=(240,240,3))) # input layer
    model.add(MaxPool2D(pool_size=(2,2),padding='same')) # padding added
    model.add(Conv2D(128, kernel_size=3, activation='relu', strides=(2, 2),activity_regularizer=regularizers.l2(0.001)))
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(Flatten()) # allowing convolution layers to be connected to fully conencted layers
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1,activation='sigmoid'))
    adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # tuning optimizer to be used
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model_smile.h5', monitor='val_loss', save_best_only=True)] # early call back function stops training when improvement is small
    history = model.fit_generator(
        X,
        steps_per_epoch= X.n/X.batch_size, # amount of data inputed at a time
        epochs=100,
        validation_data= Y,
        validation_steps= Y.n/Y.batch_size,
        callbacks= callbacks
    )
    model.summary() # provides summary of network design
    '''
    model was saved and testing in another script
    '''
    #plotting the loss and accuracy after training
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc = 'upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'],loc = 'upper left')
    plt.show()

def greying():
    '''
    Method used to greyscale images , when testing if grey scale effects results and saving them into the grey folder
    '''
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    data = np.random.random((1000, 100))
    labels_file = open('D:\\Documents\\Assignment1\\Lab21\\venv\\list.csv', 'r')
    lines = labels_file.readlines()
    all_labels = {line.split(',')[0]: int(line.split(',')[3]) for line in lines[2:]}
    labels = {}
    i = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name = img_path.split('.')[0].split('\\')[-1]
        if (all_labels[file_name] == 1):
            file_name = file_name + '.png'
            cv2.imwrite(os.path.join('D:\\Documents\\Assignment1\\Lab21\\venv\\Grey_Train\\smile',file_name),gray )
        else:
            file_name = file_name + '.png'
            cv2.imwrite(os.path.join('D:\\Documents\\Assignment1\\Lab21\\venv\\Grey_Train\\No_smile',file_name), gray)




# def train():
#
def main():
    batch_size = 50 #inputting 50 images at a time
    #agumentation of training and validation data
    train_datagen = ImageDataGenerator(rescale=1./255,#normailising image RGB values
                                       validation_split= 0.5,# spliting data 50% for training 50% for validation
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.1
                                       )

    train_generator = train_datagen.flow_from_directory(#formats images to be inputted into network for training
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train',
        subset='training',
        target_size = (240,240), # keeps original image size
        batch_size=batch_size,
        class_mode='binary' # binary classification problem


    )

    validation_generator = train_datagen.flow_from_directory(#formats images to be inputed into network for valiation
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train',
        subset='validation',
        target_size=(240,240),
        batch_size=batch_size,
        class_mode='binary'
    )
    #timing of network to execute and running netwokr
    start = time.time()
    mlp(train_generator,validation_generator)
    end = time.time()
    print("Took:")
    print((end-start)/60)
main()
