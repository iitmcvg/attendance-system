# Face Recognition

Computer Vision and Intelligence Group, IIT Madras

![avatar](g3docs/avatar.png)

We implement an experimental setup with face detection and recognition. This has been used for our purposes with the following aims:

* Swapping multiple detectors and feature extractors for facenet.
* Multi GPU and distributed support
* Frozen graph support with quantisation.

Primarily, we use this in two use cases:

* High accuracy: SSD or FRCNN detectors with Inception-Resnet feature extractors.
* CPU optimised FPS: SSDlite mobilenet with mobilenet V2 extractors (this is covered in *getting started*).

We have experimented with multiple classifiers for error free student identification.

## Sample Results

![example](media/example.jpg)

## Contents

* [Getting Started](g3docs/getting-started.md)
* [Model Zoo](g3docs/model-zoo.md)
* [Installing Dependencies](g3docs/installing_dependencies.md)

### Facenet Docs

* [David Sandberg's Implementation](g3docs/facenet.md)
* [Retraining Facenet](g3docs/facenet_train_pipeline.md)
* [Recognition with Facenet](g3docs/facenet_recognition_pipeline.md)
* [Facenet Wiki](g3docs/facenet)

### Object Detection Experimental Setup

* [Object Detection Pipeline](g3docs/object_detection_pipeline.md)

## To Do

* [ ] TF-Estimator based scalable train file.
* [x] SSDLite based detector
* [x] Mobilenet models for facenet
* [x] Angular, Focal and triplet losses.
* [ ] Inference on Singular Videos.
* [ ] DALI, Tensor RT for faster inference.
* [ ] S3D support for detection.
* [ ] Experiments with weight tying.
* [ ] Results Section
* [ ] Take a look at https://github.com/alexattia/ExtendedTinyFaces for large scale face detection.

## Dependencies

* Python 3.4+
* Tensorflow 1.7+
* Opencv 3.3.1+

## Pipeline
Image -> FaceDetection -> CroppedFace -> FaceRecognition -> Descriptor(128D) -> FaceClassifier -> Name

## Credits

## FaceRecognition(FaceNet)

TensorFlow implementation of the face recognizer described in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering". 
Ref. [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)


