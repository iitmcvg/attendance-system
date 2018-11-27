from keras.preprocessing.image import ImageDataGenerator
import numpy as  np 
import cv2
import tensorflow as tf 
import os
from skimage import transform


#datagen = ImageDataGenerator(rotation_range=20,horizontal_flip=True,brightness_range=[0.5,1],zoom_range=0.2)

path='source_path'

save_path='dest_path'




i=0
'''datagen=ImageDataGenerator(rotation_range=40,width_shift_range=2,height_shift_range=2,horizontal_flip=True,shear_range=0.1,zoom_range=0.2)
generator=datagen.flow_from_directory(path,save_to_dir=path+'/resized',save_format='jpeg',batch_size=10) '''
for filename in os.listdir(path):
    path_1=path+filename
    print(path_1)
    img=cv2.imread(os.path.join(path,filename))
    print(i)
    if img is not None:
        i+=1
        '''if i==5 :
            break'''
        flip_1 = np.fliplr(img)
        rot_1 = transform.rotate(img, angle=20, mode='reflect',preserve_range=True).astype(np.uint8)
        rot_2 = transform.rotate(img, angle=30, mode='constant',preserve_range=True).astype(np.uint8)
        rot_3 = transform.rotate(flip_1, angle=20, mode='constant',preserve_range=True).astype(np.uint8)
        cv2.imwrite(path+'/rot_1_%d.jpeg'%i,rot_1)
        cv2.imwrite(path+'/rot_2_%d.jpeg'%i,rot_2)
        cv2.imwrite(path+'/rot_3_%d.jpeg'%i,rot_3)
        cv2.imwrite(path+'/flip_1_%d.jpeg'%i,flip_1)
