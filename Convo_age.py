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

global basedir, image_paths, target_size,images_dir,labels_file
images_dir = 'D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces'

def format_data():
    '''
    splitting data into folders containong only young images and folder containing only old images
    '''
    features = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open('D:\\Documents\\Assignment1\\Lab21\\venv\\list.csv', 'r')
    lines = labels_file.readlines()
    all_labels = {line.split(',')[0]: int(line.split(',')[4]) for line in lines[2:]} # access to coloumn containing age images
    labels={}
    i=0
    for img_path in image_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        if(all_labels[file_name] == 1):
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_age\\young')
        else:
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_age\\old')

def mlp(X,Y):#input train and validation generators
    model = Sequential()#initalise model
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(2, 2), input_shape=(240, 240, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', strides=(2, 2)))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),# callback for early stop when model stops improving
                 ModelCheckpoint(filepath='best_model_age.h5', monitor='val_loss', save_best_only=True)]
    history = model.fit_generator( # trianing model by selecting batch size number of epochs
        X,
        steps_per_epoch=X.n / X.batch_size,
        epochs=100,
        validation_data=Y,
        validation_steps=Y.n / Y.batch_size,
        callbacks=callbacks
    )
    model.save_weights('first_try.h5')
    model.summary()
    #plot accuracy and loss after training
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def main():
    batch_size = 50
    #image augmentations and normalisations
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       validation_split= 0.5,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.1
                                       )

    train_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_age',
        subset='training',
        target_size = (240,240),
        batch_size=batch_size,
        class_mode='binary'


    )

    validation_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_age',
        subset='validation',
        target_size=(240,240),
        batch_size=batch_size,
        class_mode='binary'
    )
    #timing and running network
    start = time.time()
    mlp(train_generator,validation_generator)
    end = time.time()
    print((end-start)/60)
main()