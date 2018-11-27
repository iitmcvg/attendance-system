from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np 
from glob import glob
import sys
import os
from mtcnn.mtcnn import MTCNN

FACE_SIZE = 160
def align_face(image, l_eye, r_eye, desiredLeftEye = (0.35,0.35),desiredFaceWidth = FACE_SIZE, desiredFaceHeight = FACE_SIZE):
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist
    eyesCenter = tuple((np.array(l_eye) + np.array(r_eye))//2)
    print (eyesCenter)
    print (angle)
    print (scale)
    M = cv2.getRotationMatrix2D(tuple(eyesCenter), angle, scale)
    tX = desiredFaceWidth*0.5
    tY = desiredFaceHeight*desiredLeftEye[1]
    M[0,2] += (tX - eyesCenter[0])
    M[1,2] += (tY - eyesCenter[1])
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return output

detector = MTCNN()


def gamma_factor(img ,gamma):
    """
    Uses Gamma factor Power Transform to enhance image quality
    Parameters
    image: Input image
    gamma: The required exponent to perform the transformation
    Returns
    output image with the transform applied
    """
    invGamma = 1/gamma
    table_gamma =  np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(img_lab)
    channels[0] = cv2.LUT(channels[0], table_gamma )
    img_lab = cv2.merge(channels)
    img_bgr_log = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return img_bgr_log

datagen = ImageDataGenerator(
    rotation_range = 5,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.05,
    zoom_range = 0.2, 
    horizontal_flip = True,
    vertical_flip = False, 
    fill_mode = 'constant')
dataset = glob(sys.argv[1] + "/*")
j = 0
img = []
for x in dataset:
    image = (cv2.imread(x))
    image = cv2.resize(image, (800,800))
    img.append(image)
img = np.array(img)
base, classes = os.path.split(sys.argv[1])
base_data, dataset_dir = os.path.split(base)
path_new = base_data + "/" + "dataset_aug/" + classes
os.makedirs(path_new, exist_ok = True)
for batch in datagen.flow(img, batch_size = 1, seed= 1337):
    face = detector.detect_faces(batch[0])
    for idx,f in enumerate(face):
        b = f['box']
        l_eye = np.array(f['keypoints']['left_eye'])
        r_eye = np.array(f['keypoints']['right_eye'])
        face_crop_align = align_face(batch[0],l_eye, r_eye)
        cv2.imwrite(path_new+"/"+str(j) + ".jpg", face_crop_align)
        j = j + 1
        if j > 50:
            sys.exit("Done")