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
    #splits data into folders containing images with glasses and images without classes using labels from the list.csv file
    features = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open('D:\\Documents\\Assignment1\\Lab21\\venv\\list.csv', 'r')
    lines = labels_file.readlines()
    all_labels = {line.split(',')[0]: int(line.split(',')[2]) for line in lines[2:]} # column containing glassess label for all images
    labels={}
    i=0
    for img_path in image_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        if(all_labels[file_name] == 1):
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_glass\\glasses')
        else:
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_glass\\no_glasses')

def mlp(X,Y):#training an validation generator
    model = Sequential()#initalise model
    model.add(Conv2D(64,kernel_size= 3,activation='relu',strides=(2,2),input_shape=(240,240,3))) # input layer
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))#padding added
    model.add(Conv2D(128, kernel_size=3, activation='relu', strides=(2, 2),activity_regularizer=regularizers.l2(0.001)))#regulization added to convolutional layer with lamda parameter 0.001
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(Flatten())#allow connection to fukky connect layer
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1,activation='sigmoid'))
    adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # optimzer created
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])#back propogation
    callbacks = [EarlyStopping(monitor='val_loss', patience=100),
             ModelCheckpoint(filepath='best_model_glasses.h5', monitor='val_loss', save_best_only=True)]#conditon for early stop when model stops improving
    history = model.fit_generator(
        X,
        steps_per_epoch= X.n/X.batch_size,
        epochs=100, # Maximum number of forward + back props
        validation_data= Y,
        validation_steps= Y.n/Y.batch_size,
        callbacks= callbacks
    )
    model.save_weights('first_try.h5')
    model.summary()
    #plots training graphs after trianing (accuracy and loss achieved)
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

def main():
    batch_size = 50
    #dataset augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,#normalised
                                       validation_split= 0.5,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.1
                                       )

    train_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_glass',
        subset='training',
        target_size = (240,240),
        batch_size=batch_size,
        class_mode='binary'


    )

    validation_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_glass',
        subset='validation',
        target_size=(240,240),
        batch_size=batch_size,
        class_mode='binary'
    )
    #timing and running network
    start = time.time()
    mlp(train_generator,validation_generator)
    end = time.time()
    print((end-start)/2)
main()