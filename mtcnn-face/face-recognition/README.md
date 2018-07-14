# MTCNN - face detection

Follow the installation procedures given in the Facenet repo to install the modules required. Download the pretrained model from the repo.

Have a dataset with different faces in different sub-directories.

Training
--------

```bash
python facenet/src/custom_face_detection.py TRAIN <dataset_folder>  <path to .pb file>  <path for svm .pkl file> --batch_size 16 --min_nrof_images_per_class 40
```

Classification
---

```bash
python facenet/src/custom_face_detection.py CLASSIFY <dataset_folder>  <path to .pb file>  <path for svm .pkl file> --batch_size 16 --min_nrof_images_per_class 40
```


Live face recognition
---

```bash
python facenet/src/custom_face_detection.py LIVE dataset_aug   models/20180402-114759.pb  models/lfw_classifier.pkl 
```

Datset Creation
---

This code can be used to create the dataset. Enter the name of the participant and live window appears. Press `s` to save the image in the directory.

```bash
python dataset_creation.py
```

Data Augmentation
---

Enter the path of the folder of images you would like to augment.
Default destination directory will be `dataset_aug`

```bash
python data_augmentation.py <path to the  folder of images>
```