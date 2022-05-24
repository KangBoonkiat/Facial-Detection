# Facial-Detection

### Few libraries need to install:
                                1. opencv-python 
                                2. tensorflow 
                                3. tensorflow-gpu
                                4. labelme
                                5. albumentations
                                6. numpy

Nvidia GPU user - install cuda and cudnn 


### 1.Data Collection and sloting:

1. Create own dataset using webcam, make use of the DataCollection.py
2. Moving of randam amount and image data from a folder to other folder using movefile.py

### 2.Create data annotation:

1. After installing labelme, go to the enviroment that labelme is installed and type labelme.
2. A program should pop out for you to create ur own annotation

If you need to move the created annotations to the right folder, use the movelabel.py

### 3.Create more data:

1. Make use of the create_data.py
2. Kindly ensure that the folder for image and annotations are set correct if not it will not work

### 4.Build Model:

1. The Model used will be VGG16.
2. In BuildModel.py, it is all the step to build model and save it. 

### 5.Testing Model
1. In live_detection.py, is all step to load the saved model and run it.


## Common Problem and Solution

### 1. If you can't activate the webcam: 
1. Ensure the webcam is not in-use
2. pip uninstall opencv and pip install opencv-python

### 2. If you have issue while using code that require tensorflow:
1. check if tensorflow and tensorflow-gpu is compactable with the python version you using.https://www.tensorflow.org/install/source#gpu 
2. Ensure tensorflow and tensorflow-gpu are the same version

### 3. For Nvidia gpu user: Error on cudnn64_8.dll
1. Ensure cudnn is installed probably.
2. Go to cuddnn/bin folder and copy all the cudnn_xx_xxxx64_8.dll
3. Paste it at cuda/bin
                        
