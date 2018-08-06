'''
CVI, IITM 2018

Train SVM or KNN classifier for face recognition.

Args:
* detection model: detection model to use. 
    SSDMobilenet and SSDlite_Mobilenet_v2 (default) are the avaliable choices. 
    The latter is more accurate and yields larger FPS.
    See training details for descriptions on mAP and metrics.

* recognition model: recognition model to use. 
    SSDMobilenet and SSDlite_Mobilenet_v2 (default) are the avaliable choices. 
    The latter is more accurate and yields larger FPS.
    See training details for descriptions on mAP and metrics.

* classifier: whether to use KNN or SVM classifier.

* embedding_size: Change for non-standard configuration. If not, leave as it is.

* trained_clasifier: file for trained classifier to be stored.

Example:

python demo_face_recognition.py 
--detection_model ssdlite_v2
--recognition_model mobilenet_v2
--camera 1
'''

import cv2
import os
import numpy as np
from recognition.facenet import get_dataset
from recognition.FaceRecognition import FaceRecognition
from detection.FaceDetector import FaceDetector
from classifier.FaceClassifier import FaceClassifier
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--recognition_model",default="mobilenet_v2",
choices=["inception_resnet_v1","mobilenet_v2","inception_resnet_v2"])

parser.add_argument("--detection_model",default="ssdlite_v2",
choices=["ssdlite_v2","ssd_mobilenet"],help="detection model to use")

parser.add_argument("--trained_classifier",default="./classifier/trained_classifier.pkl"
,help="trained classifier to use")

parser.add_argument("--classifier",default="SVM",choices=["SVM","random-forests","KNN","DNN"]
,help="trained classifier to use")

parser.add_argument("--embedding_size",default=512,choices=[128,512]
,help="Embedding Size")

datadir = './media/train_classifier'

args=parser.parse_args()

if args.detection_model=="ssdlite_v2":
    detect_ckpt = 'model/ssdlite_v2.pb'
elif args.detection_model=="ssd_mobilenet":
    detect_ckpt = 'model/ssd_mobilenet.pb'

if args.recognition_model=="inception_resnet_v1":
    recog_ckpt = 'model/inception_resnet-20180402-114759.pb'
elif args.recognition_model=="mobilenet_v2":
    recog_ckpt = 'model/mobilenet_v2.pb'
elif args.recognition_model=="inception_resnet_v2":
    recog_ckpt = 'model/inception_resnet_v1_20170512-110547.pb'

if args.classifier=="SVM":
    model_type="SVM"
elif args.classifier=="random-forests":
    model_type="random-forests"
elif args.classifier=="KNN":
    model_type="knn"
elif args.classifier=="DNN":
    model_type="dnn"

if args.recognition_model=="mobilenet_v2":
    mobilenet=True

face_detector = FaceDetector(PATH_TO_CKPT=detect_ckpt)
face_recognition = FaceRecognition(PATH_TO_CKPT=recog_ckpt)
face_classfier = FaceClassifier(args.trained_classifier)

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [dataset[i].name] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

dataset = get_dataset(datadir)
paths, labels = get_image_paths_and_labels(dataset)
print('Number of classes: %d' % len(dataset))
print('Number of images: %d' % len(paths))

# Run forward pass to calculate embeddings
print('Calculating features for images')
image_size = 160
nrof_images = len(paths)
features = np.zeros((2*nrof_images, args.embedding_size))
labels = np.asarray(labels).repeat(2)
for i in range(nrof_images):
    img = cv2.imread(paths[i])
    if img is None:
        print('Open image file failed: ' + paths[i])
        continue
    boxes, scores = face_detector.detect(img)
    if len(boxes) < 0 or scores[0] < 0.5:
        print('No face found in ' + paths[i])
        continue

    cropped_face = img[boxes[0][0]:boxes[0][2], boxes[0][1]:boxes[0][3], :]
    cropped_face_flip = cv2.flip(cropped_face,1)
    features[2*i,:] = face_recognition.recognize(cropped_face,mobilenet=mobilenet)
    features[2*i+1,:] = face_recognition.recognize(cropped_face_flip,mobilenet=mobilenet)

print('Start training for images')
face_classfier.train(features, labels, model=model_type, save_model_path=args.trained_classifier)
print('Finished training.')
