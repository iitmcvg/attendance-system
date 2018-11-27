# Detection
We have experimented with different detection models with validation on the hard set of WIDER dataset (S.O.T.A mAP is 46.4) :

## Experimenting with detection models

| Model  | mAP | Iterations trained | 
| ------------- | ------------- |
| SSDlite mobilenet v2|  32.7 |  100k |
| SSDlite mobilenet |  33.4 |  110k |
| SSDlite mobilenet with scaling and interpolation|  38.6 |  110k |
| FRCNN Resnet50| 40.8 |  120k |
| FRCNN inception v3 | 43.9 | 120k|

# Recoginition
 Faster R-CNN Resnet50 model has been used for recognition.
## Data augmentation
To prevent the model from overfitting, we implemented multiple data augmentation techniques :
* Flips
* Rotations
* Brightness adjustments: Images of different brightness levels were generated using the Gamma correction/Power Law transform.

## Classification
We first extracted facenet embeddings for a dataset of 1110 images with 8 classes.
We have mainly experimented with the following classifiers :
* SVMs
* K-Nearest Neighbours
* Random forests
* Classification using L2

We used of ensemble of SVMs (RBF,linear and poly), K-NNs ( k=5 and k=7) and random forests for the final output.

