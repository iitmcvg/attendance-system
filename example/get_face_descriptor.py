import cv2
import numpy as np
from recognition.FaceRecognition import FaceRecognition
from detection.FaceDetector import FaceDetector

face_detector = FaceDetector()
face_recognition = FaceRecognition()
image_files = ['./media/1.jpg', './media/2.jpg']
for input_str in image_files:

    img = cv2.imread(input_str)
    boxes, scores = face_detector.detect(img)
    face_boxes = boxes[np.argwhere(scores>0.5).squeeze()]
    print('Number of face in image:', len(face_boxes))
    for box in face_boxes:
        cropped_face = img[box[0]:box[2], box[1]:box[3], :]
        cropped_face = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)

        print('Face descriptor:')
        print(face_recognition.recognize(cropped_face), '\n')
        cv2.imshow('image', cropped_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
