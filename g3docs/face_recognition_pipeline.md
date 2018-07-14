# Face Recognition Pipeline

## Adding users

```
python train_face_classify.py
```

takes images under `media/train_classifier/` with each folder as a subject. Add all subject photos in the same folder.

## Running recognition

```
python demo_face_recognition.py
```

Uses cv2.Videocapture(0) as default. Non-Queue version

## Running detection only.

```
python demo_face_detect.py
```

Uses cv2.Videocapture(0) as default. Non-Queue version