# Open-CV-traffic-Light-recognation

## version 1 

Testing one single image to recognize traffic light
Using OpenCV (cv2) for detecting traffic light colors

Folder name: 
Problem: 
1. when there is no traffic light  it will find the most color inside the image
2. if there are many traffic lights inside the image, focus on the left side of the traffic light as the final result


## version 2

following this sample https://cloudinary.com/guides/image-effects/building-a-python-image-recognition-system
Building a Python Image Recognition System

Using TensorFlow to load the ResNet50 model for classification and OpenCV (cv2) to read and process images from a directory and  for color detection.


Problem:
1. This training model is trained to recognize football images. The result is not accurate.


## Version 3

Following this sample [https://www.kaggle.com/code/meemr5/traffic-light-detection-pytorch-starter#5.-Validation-Scheme](https://www.kaggle.com/code/photunix/classify-traffic-lights-with-pre-trained-cnn-model?utm_source=chatgpt.com)
Classify traffic lights with a pre-trained CNN model

trainmodel_New.py and trainmodel1 copy.py is the original code. The main differenceis  one with a description and one is not.
trainmodel.py find a  traffic light in one image and crop it.
Detect_Traffic_light.py is to detect multiple image with traffic light
Circle_Traffic_light.py is find and Circle the traffic light
draw_rectangles_on_Traffic_light.py is find and draw a rectangle to the traffic light.


Problem solve.
1. Able to find the traffic light correctly, even if the image has multiple traffic in it
