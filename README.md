# Assignment1
First run Assignment.py

>This code is an adapted value of lab2 and uses the libraries numpy,cv2,os,shutil,keras and dlib
>execute this code by running it
>This should create the directory nonface

Running Convo_smile
>install libraries numpy,cv2,shutil,keras,dlib,imutils,sklearn,matplotlib,time,pandas
>First run format_data() making sure that image directory is set to nonface and the label file is set to the label spreadsheet
>The train_generator and validation_generator directories should be alter to where the file is present
>run main()


This is the case for all binary task 

smile_classification_MLP.py
used to run all mlp models
>change directories within the train and validation generators to contain the folder containing the classes of the problem formed by the format_data() function in the other conovloution scripts
>run
Running Convo_hair.py
>>install libraries numpy,cv2,shutil,keras,dlib,imutils,sklearn,matplotlib,time,pandas
>First run format_data() making sure that image directory is set to nonface and the label file is set to the label spreadsheet
>remove unknown from the directory manually
>The train_generator and validation_generator directories should be alter to where the file is present
>run main()

Running testing.py

>After trainning the network the model is saved This model should be loaded in by entering the correct directories into line 24
>The train_generator and test_generator directories should be alter to where the file is present

Running augmentation.py
>Change the image read path to the path where the image exsist
>run


The format data should create these directories for these classification problems :

Train - smile,
Train_age - agem
Train_human - human,
Train_glasses - glasses,
Train_hair - hair,
