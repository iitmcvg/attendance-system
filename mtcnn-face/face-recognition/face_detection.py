from mtcnn.mtcnn import MTCNN
import cv2

cap = cv2.VideoCapture(0)
detector = MTCNN()
while True:
    ret, frame = cap.read()
    face = (detector.detect_faces(frame))
    if len(face) > 0:
        for f in face:
            b = f['box']
            cv2.rectangle(frame, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,255,0))
    cv2.imshow("iamge", frame)
    cv2.waitKey(10)