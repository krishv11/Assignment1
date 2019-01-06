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
from keras.models import load_model

model = load_model('D:\\Documents\\Assignment1\\Lab21\\venv\\best_model_hair.h5') #load the model to be trained from the saved file

batch_size = 50
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.5,
                                   horizontal_flip=True,
                                   shear_range=0.2,
                                   zoom_range=0.1
                                   )
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  )#test image augmentation

train_generator = train_datagen.flow_from_directory(
    'D:\\Documents\\Assignment1\\Lab21\\venv\\Train_hair',
    subset='training',
    target_size=(240, 240),
    batch_size=batch_size,
    class_mode='binary'

)

validation_generator = train_datagen.flow_from_directory(
    'D:\\Documents\\Assignment1\\Lab21\\venv\\Train',
    subset='validation',
    target_size=(240, 240),
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(#train input
    'D:\\Documents\\Assignment1\\Lab21\\venv\\testing_dataset',
    target_size=(240, 240),
    batch_size=1,
    class_mode=None, # as no labels
    shuffle=False,
)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#reset the generator
test_generator.reset()
pred = model.predict_generator(test_generator,steps=100)#making the predictions , outputting the probability of each probability
print(pred)
predicted_class_indices=pred.argmax(axis = -1) #this line for multiclass output the class value 0-5
# predicted_class_indices=np.rint(pred) # commnt out above line and uncomment this line for binary classification rounds probabilities to 0 and 1
print(predicted_class_indices)
labels= (train_generator.class_indices) # the labels for the classes
print(labels)
## putting the imges and prediction into an CSV file.
filenames = test_generator.filenames
print(filenames)
results = pd.DataFrame({"Filename":filenames,
                            "Predictions":predicted_class_indices.flatten()})
results.to_csv("results_hair.csv",index = False)
