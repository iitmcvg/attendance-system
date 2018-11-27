

from keras.preprocessing.image import ImageDataGenerator
import imgaug
import numpy as  np 
import cv2
import tensorflow as tf 
import os
from skimage import transform

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 

	return cv2.LUT(image, table)


path='add source file path here'
path2='add destination file path here'

i=0
for filename in os.listdir(path):
    original=cv2.imread(os.path.join(path,filename))
    if original is None :
        break
    original=cv2.resize(original,(400,400))
    for gamma in np.arange(0.5, 2, 0.5):
        # ignore when gamma is 1 (there will be no change to the image)

        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(original, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(path2,'{}_gamma_{}.jpeg'.format(i,gamma)),adjusted)
        i+=1
        #cv2.imshow("Images", adjusted)
        #cv2.waitKey(0)


