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

def mlp(X,Y):
    #the mlp network
    model = Sequential()
    model.add(Dense(32, input_shape =(240,240,3)  ,activation= 'relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu',input_dim=6))
    model.add(Dense(6,activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),#early stop if network improvement not occuring
                 ModelCheckpoint(filepath='best_model_MLP_hair.h5', monitor='val_loss', save_best_only=True)]
    history = model.fit_generator(#training network
        X,
        steps_per_epoch=X.n / X.batch_size,
        epochs=30
        ,
        validation_data=Y,
        validation_steps=Y.n / Y.batch_size,
        callbacks=callbacks
    )
    model.save_weights('first_try.h5')
    #prinining accurary and loss of training
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
    greying of dataset for smile however can be changed to other datasets by altering lines 78 and the folder to move it to
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
    #batch size reduced to the the network eing slower
    batch_size = 35
    #image augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       validation_split= 0.5,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.1
                                       )
    test_datagen = ImageDataGenerator(rescale=1./255,
                                      )

    train_generator = train_datagen.flow_from_directory(#In this case hair classification was being done however by changing the directiory the ohter task were also test
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_hair',
        subset='training',
        target_size = (240,240),
        batch_size=batch_size,
        class_mode='categorical'


    )

    validation_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_hair',
        subset='validation',
        target_size=(240,240),
        batch_size=batch_size,
        class_mode='categorical'
    )
    #timing and running network
    start = time.time()
    mlp(train_generator,validation_generator)
    end = time.time()
    print((end-start)/60)
main()

