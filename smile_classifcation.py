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

global image_paths
shape_predictor="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
# face_cascade = cv.CascadeClassifier('D:\\Documents\\Assignment1\\Lab21\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_smile.xml')
# print(face_cascade.load('D:\\Documents\\Assignment1\\Lab21\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_smile.xml'))
# img = cv.imread('D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces\\291.png')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(
# gray, scaleFactor=1.7, minNeighbors=3, minSize=(15, 15))
# print (format(len(faces)))
#
# for (x,y,w,h) in faces :
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,0,0),5)
#
# cv.imshow('img',img)
# cv.waitKey(0)
# # cv.imshow("Facesss",img)
# cv.waitKey(0)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def smile_distance(img_path):

    img = cv2.imread(img_path)

    img = imutils.resize(img,width=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Frame", img)
# cv2.waitKey(0)
    rects = detector(gray, 0)
    dist_smile = 0
    j =0

    for rect in rects:
        shape = predictor(gray,rect)
        shape = shape_to_np(shape)


        x49 = shape[48,0]
        y49 = shape[48,1]

        x55 = shape[54,0]
        y55 = shape[54,1]

        dist_smile = ((x49 - x55) ** 2 + (y49 - y55) ** 2) ** 0.5


    return dist_smile
'''
        print(shape[48,48])
        print(shape[54,54])
        for (x, y) in shape:
            # cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            if (i == 49):
                x49 = x
                y49 = y
                print(x)
            if (i == 55):
                x55 = x
                y55 = y
                print(y)
                dist_smile = ((x49-x55)**2 + (y49-y55)**2)**0.5
            i = i + 1

'''

def train_SVM(training_images, training_labels, test_images, test_labels):
    #get_data()
    clf= svm.SVC(gamma='scale',kernel='rbf')
    clf.fit(training_images,training_labels)
    y_pred = clf.predict(test_images)
    # print("Confusion")
    # print( confusion_matrix(test_labels, y_pred))
    print("Classification report")
    print(classification_report(test_labels, y_pred))
    return 0

def get_labels(all_labels):
    labels={}
    images_dir = 'D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces'
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    for img_path in image_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        labels[file_name] = all_labels[file_name]
    return labels

# x = np.array([[,]])
# print(x)
# x = np.append(x,[[dist_smile,smile]],axis=0)
# # print(x)
# print(np.shape(x))

images_dir = ('D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces')
labels_file = open('D:\\Documents\\Assignment1\\Lab21\\venv\\list.csv', 'r')
lines = labels_file.readlines()
smile = {line.split(',')[0] : int(line.split(',')[3]) for line in lines[2:]}
labels = get_labels(smile)
target_size = None
image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
i = 1
if os.path.isdir(images_dir):
    all_features = []
    all_labels = []
    for img_path in image_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        print(i)
        i = i + 1
        distance = smile_distance(img_path)
        all_features.append(distance)
        all_labels.append(labels[file_name])
    smile_feature = np.array(all_features)
    smile_labels = ((np.array(all_labels)+1)/2) # -1 = 0 1 = 1
#
X = smile_feature
Y = smile_labels
# random_30 = 1310
# index = np.random.choice(X.shape[0], random_30, replace=False)
tr_X = X[:2184]
tr_Y = Y[:2184]
te_X = X[2184:]
te_Y = Y[2184:]
#
plt.plot(tr_X,tr_Y,'ro')
#
plt.show()
# tr_X = np.reshape(tr_X,(-1,1))
# print(tr_X.shape)
# te_X = np.reshape(te_X,(-1,1))
# train_SVM(tr_X,tr_Y,te_X,te_Y)

# print(smile_feature)
