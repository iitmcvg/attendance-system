'''
CVI, IITM 2018

Register a new face.

Pipeline:
Detect > Crop > Embeddings > Cluster to recognise.

Args:
* detection model: detection model to use. 
    SSDMobilenet and SSDlite_Mobilenet_v2 (default) are the avaliable choices. 
    The latter is more accurate and yields larger FPS.
    See training details for descriptions on mAP and metrics.

* recognition model: recognition model to use. 
    SSDMobilenet and SSDlite_Mobilenet_v2 (default) are the avaliable choices. 
    The latter is more accurate and yields larger FPS.
    See training details for descriptions on mAP and metrics.

* camera: which port to use for opencv's Videocapture object. 
    0 is the usual for webcameras, USB cameras are listed as 1,2...

Example:

python demo_face_recognition.py 
--detection_model ssdlite_v2
--recognition_model mobilenet_v2
--camera 1
'''

import cv2
import time
import numpy as np
from detection.FaceDetector import FaceDetector
from recognition.FaceRecognition import FaceRecognition
from classifier.FaceClassifier import FaceClassifier
import argparse
import os
parser=argparse.ArgumentParser()
import skimage.io as io
parser.add_argument("--recognition_model",default="mobilenet_v2",
choices=["inception_resnet_v1","mobilenet_v2","inception_resnet_v1_old"])

parser.add_argument("--detection_model",default="ssdlite_v2",
choices=["ssdlite_v2","ssd_mobilenet"],help="detection model to use")

parser.add_argument("--trained_classifier",default="./classifier/trained_classifier.pkl"
,help="trained classifier to use")

parser.add_argument("--attendance_file",default="./attendance.txt"
,help="Output attendance file to use.")

parser.add_argument("--attendance_csv",default="attendance.csv"
,help="Output attendance file to use.")

parser.add_argument("--clear",default=""
,help="Clear attendance or not.")


# Camera
parser.add_argument("--camera",default=0,
type=int,help="Camera to use | 0 webcam | 1 usb camera. Can be different if lack of drivers.")

args=parser.parse_args()

if args.detection_model=="ssdlite_v2":
    detect_ckpt = 'model/ssdlite_v2.pb'
elif args.detection_model=="ssd_mobilenet":
    detect_ckpt = 'model/ssd_mobilenet.pb'

if args.recognition_model=="inception_resnet_v1":
    recog_ckpt = 'model/inception_resnet-20180402-114759.pb'
elif args.recognition_model=="mobilenet_v2":
    recog_ckpt = 'model/mobilenet_v2.pb'
elif args.recognition_model=="inception_resnet_v1_old":
    recog_ckpt = 'model/inception_resnet_v1_20170512-110547.pb'
mobilenet=False
if args.recognition_model=="mobilenet_v2":
    mobilenet=True

def _update_attendance(attendance_list):
    '''
    Updates attendance

    Args:
    attend

    '''
    with open(args.attendance_file,"w") as f:
        attendance_list=[a+"\n" for a in attendance_list]
        f.writelines(attendance_list)

face_detector = FaceDetector(PATH_TO_CKPT=detect_ckpt)
face_recognition = FaceRecognition(PATH_TO_CKPT=recog_ckpt)
face_classfier = FaceClassifier(args.trained_classifier)
#video_capture = cv2.VideoCapture(args.camera)

print('Start Recognition!')
prevTime = 0
count=0
attendance_list=[]
i=1
while True:
    j=37+i
    path = './media/train_classifier/milind/'
    frame= cv2.imread(path+'WhatsApp Image 2018-11-01 at 9.23.%d PM.jpeg'%j)
    print('%dth recognition'%i)
    i+=1
    
    #if np.size(frame)==1:
    #    continue
    frame = cv2.resize(frame, (600,600))  # resize frame (optional)
    '''cv2.imshow('frame',frame)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()'''
    curTime = time.time()  # calc fps
    find_results = []

    frame = frame[:, :, 0:3]
    boxes, scores = face_detector.detect(frame)
    #print(scores)
    face_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)] 
    face_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
    print('Detected_FaceNum: %d' % len(face_boxes))

    print('Registering only the maximum score face.')
    name='milind'
    box_ii=np.argmax(face_scores)
    print(box_ii)
    path=os.path.join("media","train_classifier",'milind')
    attendance_list.append(name)
    _update_attendance(attendance_list)
    #print(type(path))
    if not os.path.exists(path):
        os.mkdir(path)
   
    
    
    if len(face_boxes) > 0:
        box = face_boxes[box_ii]
        cropped_face = frame[box[0]:box[2], box[1]:box[3], :]
        cropped_face = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)

        feature = face_recognition.recognize(cropped_face,mobilenet=mobilenet)
        print("len features {}".format(len(feature)))
        name_features = face_classfier.classify(feature)
        #madeachange
        print('name features',name_features)
        if os.path.exists(os.path.join("media","train_image_classifier",name)):
            continue
        # Expand by 20 pixels
        save_face=frame[box[0]-20:box[2]+20, box[1]-20:box[3]+20, :]
        #print(type(save_face))
        #cv2.imwrite(os.path.join(path,name+"_"+"%d"%(count)+".jpg"),save_face)

        count+=1

        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

        # plot result idx under box
        text_x = box[1]
        text_y = box[2] + 20
        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 0, 255), thickness=1, lineType=2)
    else:
        print('Unable to align')

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = 'FPS: %2.3f' % fps
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    cv2.putText(frame, str, (text_fps_x, text_fps_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)

    cv2.imshow('Video', frame)
    cv2.imshow('cropped face',cropped_face)
    if count==16:
        break
    if i==10:
        break
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    count+=1

#video_capture.release()
cv2.destroyAllWindows()

