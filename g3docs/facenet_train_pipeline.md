# Facenet train Pipeline 

Dataset: [VGGFace 2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

`export PYTHONPATH=facenet/src`

## /media/ssd1/face/datasets structure

Tree
```
/media/ssd1/face/dataset
|-- lfw-deepfunneled
|-- lfw_mtcnnalign
|-- test
|-- train
|-- train_mtcnnalign
```

test, train refer to VGGFace 2
[LFW](http://vis-www.cs.umass.edu/lfw/) is used for benchmarking. Deepfunneled branch used.
Total number of images: 13233 (LFW Deepfunneled).

## Dataset structure
It is assumed that the training dataset is arranged as below, i.e. where each class is a sub-directory containing the training examples belonging to that class.

```
Aaron_Eckhart
    Aaron_Eckhart_0001.jpg

Aaron_Guiel
    Aaron_Guiel_0001.jpg

Aaron_Patterson
    Aaron_Patterson_0001.jpg

Aaron_Peirsol
    Aaron_Peirsol_0001.jpg
    Aaron_Peirsol_0002.jpg
    Aaron_Peirsol_0003.jpg
    Aaron_Peirsol_0004.jpg
    ...
```

## Face alignment

```
for N in {1..8}; do \
python facenet/src/align/align_dataset_mtcnn.py \
/media/ssd1/face/dataset/train/ \
/media/ssd1/face/dataset/train_mtcnnalign \
--image_size 182 \
--margin 44 \
--random_order \
--gpu_memory_fraction 0.25 \
& done
```

### For LFW (Eval purposes)

python facenet/src/align/align_dataset_mtcnn.py \
/media/ssd1/face/dataset/lfw-deepfunneled \
/media/ssd1/face/dataset/lfw_mtcnnalign \
--image_size 182 \
--margin 44 \
--random_order \
--gpu_memory_fraction 1

## Model (checkpoint storage) structure

/

## Classifier Training (Softmax loss)

```
python facenet/src/train_softmax.py \
--logs_base_dir /media/ssd1/face/facenet/logs \
--models_base_dir /media/ssd1/face/facenet/models \
--data_dir /media/ssd1/face/dataset/train_mtcnnalign \
--image_size 160 \
--model_def models.mobilenet_v2 \
--lfw_dir /media/ssd1/face/dataset/lfw_mtcnnalign/ \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 500 \
--batch_size 90 \
--keep_probability 0.4 \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file facenet/data/learning_rate_schedule_classifier_vggface2.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.01 \
--validate_every_n_epochs 5
```

### Distance Metrics


