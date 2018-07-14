import cv2
import os
import cv2
import sys
from mtcnn.mtcnn import MTCNN
import numpy as np
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

cap = cv2.VideoCapture(0)
while True:
    print ("Enter the name of the participant")
    name = input()
    os.makedirs("dataset/" + name, exist_ok=True)
    path = "dataset/" + name
    u = 0
    while True:
        ret, frame = cap.read()
        face = (detector.detect_faces(frame))
        fc = frame.copy()
        
        if len(face) > 0:
            for idx,f in enumerate(face):
                # print (f)
                b = f['box']
                cv2.rectangle(fc, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,255,0))
                face_crop = frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
                face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
                l_eye = np.array(f['keypoints']['left_eye'])
                r_eye = np.array(f['keypoints']['right_eye'])

                # face_crop_align = align_face(fc,l_eye, r_eye)
                cv2.putText(fc, "Human Face", (b[0], b[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                # cv2.imshow("iamge", np.hstack((face_crop, face_crop_align)))
        k = cv2.waitKey(1) & 0xFF
        cv2.imshow("frame_", fc)
        if k == ord('s'):
            # print ("entered")
            cv2.imwrite(path+"/"+str(u)+".jpg",frame)
            # cv2.imwrite(path+"/"+str(u)+".jpg", frame)
            u = u + 1
            print (r"Image No:" + str(u), end = "\r")
        elif k == 27:
            cv2.destroyAllWindows()
            break