import numpy as np
import cv2
import os
from shutil import copy
from keras.preprocessing import image
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# face_cascade = cv.CascadeClassifier('D:\\Documents\\Assignment1\\Lab21\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
# img = cv.imread('D:\\Documents\\Assignment1\\Lab21\\venv\\dataset\\4.png')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray,1.1,1)
# print (format(len(faces)))
# cv.imshow("Facesss",img)
# cv.waitKey(0)
# images = []

# images_dir =('D:\\Documents\\Assignment1\\Lab21\\venv\\dataset')
# image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
# if os.path.isdir(images_dir):
#     for img_path in image_paths:
#         #x = 0
#         img = cv.imread(img_path)
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray,1.1,1)
#         if (len(faces) == 0):
#             copy(img_path,'D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces')

        #images.append(len(faces))

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

images_dir =('D:\\Documents\\Assignment1\\Lab21\\venv\\dataset')
image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
target_size = None
if os.path.isdir(images_dir):
    for img_path in image_paths:
        img = image.img_to_array(
            image.load_img(img_path,
                           target_size=target_size,
                           interpolation='bicubic'))
        features, _ = run_dlib_shape(img)
        if features is not None:
            copy(img_path, 'D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces')

print('done')
