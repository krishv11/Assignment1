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
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model




global basedir, image_paths, target_size,images_dir,labels_file
images_dir = 'D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces'

# lines = labels_file.readlines()
# labels = {line.split(',')[0] : int(line.split(',')[3]) for line in lines[2:]}
# image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
# smile_labels= {}
# list_image = ()
#
# for img_path in image_paths:
#     file_name = img_path.split('.')[0].split('\\')[-1]
#     print(file_name)
#     print(labels[file_name])
#     smile_labels[file_name] = labels[file_name]
#
# nClasses = 2
# print("Number of classes(Outputs):",nClasses)
# print(img_path)
# # img = cv2.imread(img_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# cv2.imshow("Label: {}".format(smile_labels[file_name]),img)
# cv2.waitKey(0)
#
# def get_data():
#     for img_path in image_paths:
#         file_name = img_path.split('.')[0].split('\\')[-1]
#         print(file_name)
#         print(labels[file_name])
#         img = cv2.imread(img_path)
#         smile_labels[file_name] = labels[file_name]
#         list.append(img)
#     return smile_labels


def format_data():
    features = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open('D:\\Documents\\Assignment1\\Lab21\\venv\\list.csv', 'r')
    lines = labels_file.readlines()
    all_labels = {line.split(',')[0]: int(line.split(',')[3]) for line in lines[2:]}
    labels={}
    i=0
    for img_path in image_paths:
        # img= cv2.imread(img_path)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = gray/255
        # gray = gray.ravel()
        # features.append(gray) # list of all images
        file_name = img_path.split('.')[0].split('\\')[-1]
        if(all_labels[file_name] == 1):
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train\\smile')
        else:
            copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\Train\\No_smile')
    #     print(i)
    # hi = np.vstack(features)
    # print(hi.ndim)
    # print(hi)






def mlp(X,Y):
    print("Welcome")
    model = Sequential()
    model.add(Dense(10, input_shape =(240,240,3)  ,activation= 'relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit_generator(
        X,
        steps_per_epoch= X.n/X.batch_size,
        epochs=3,
        validation_data= Y,
        validation_steps= Y.n/Y.batch_size
    )
    model.save_weights('first_try.h5')
    # soores = model.evaluate(X,Y)
    # print('\n%s: %.2f%%' % (model.metrics_names[1],soores[1]*100))


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
        #     print(i)




# def train():
#
def main():
    batch_size = 50
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       validation_split= 0.7,
                                       horizontal_flip=True,
                                       )
    test_datagen = ImageDataGenerator(rescale=1./255,
                                      )

    train_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train',
        subset='training',
        target_size = (240,240),
        batch_size=batch_size,
        class_mode='binary'


    )

    validation_generator = train_datagen.flow_from_directory(
        'D:\\Documents\\Assignment1\\Lab21\\venv\\Train',
        subset='validation',
        target_size=(240,240),
        batch_size=batch_size,
        class_mode='binary'
    )
    mlp(train_generator,validation_generator)
main()

# format_data()
# mlp(X,Y)
