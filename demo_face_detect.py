'''
CVI, IITM 2018

Demo for face detection. Runs given model specification for detecting faces.

Args:
* detection model: detection model to use. 
    SSDMobilenet and SSDlite_Mobilenet_v2 (default) are the avaliable choices. 
    The latter is more accurate and yields larger FPS.
    See training details for descriptions on mAP and metrics.

* camera: which port to use for opencv's Videocapture object. 
    0 is the usual for webcameras, USB cameras are listed as 1,2...

Example:

python demo_face_detect.py 
--detection_model ssdlite_v2
--camera 1
'''

import cv2
import time
import numpy as np
from detection.FaceDetector import FaceDetector
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--detection_model",default="ssdlite_v2",
choices=["ssdlite_v2","ssd_mobilenet"],help="detection model to use")

# Camera
parser.add_argument("--camera",default=0,
type=int,help="Camera to use | 0 webcam | 1 usb camera. Can be different if lack of drivers.")

args=parser.parse_args()

if args.detection_model=="ssdlite_v2":
    detect_ckpt = './model/ssdlite_v2.pb'
elif args.detection_model=="ssd_mobilenet":
    detect_ckpt = './model/ssd_mobilenet.pb'

face_detector = FaceDetector(PATH_TO_CKPT=detect_ckpt)
video_capture = cv2.VideoCapture(args.camera)

print('Start Recognition!')
prevTime = 0

while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)  # resize frame (optional)

    curTime = time.time()  # calc fps
    find_results = []

    frame = frame[:, :, 0:3]
    boxes, scores = face_detector.detect(frame)
    face_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
    face_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
    print('Detected_FaceNum: %d' % len(face_boxes))

    if len(face_boxes) > 0:
        for i in range(len(face_boxes)):
            box = face_boxes[i]
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            # plot result idx under box
            text_x = box[1]
            text_y = box[2] + 20
            cv2.putText(frame, 'Score: %2.3f' % face_scores[i], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

