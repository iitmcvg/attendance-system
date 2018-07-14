# README

_Task Objective:_ To detect faces in face data.

Dataset: [WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
with 32,203 images and label 393,703 faces.

---

## Folder-wise Documentation

* `g3docs`: Google provided Documentation of some cases
* `protos`: Protobufs of all config files. Refer for adding other options. Prefer adding new configuration options as flags to protobufs than code.
* `dataset_tools`: Set of TF Record creating tools.
* `dataset_decoders`: Decode a specific TF Record.

----
## Directory Structure

### Data

Tree

```
.
├── annotations
│   ├── readme.txt
│   ├── wider_face_test.mat
│   ├── wider_face_test_filelist.txt
│   ├── wider_face_train.mat
│   ├── wider_face_train_bbx_gt.txt
│   ├── wider_face_val.mat
│   └── wider_face_val_bbx_gt.txt
├── data
│   ├── WIDER_train
│   │   └── images
│   └── WIDER_val
│       └── images
├── records
└── ssdlite_v2
    ├── checkpoint
    │   ├── checkpoint
    │   ├── frozen_inference_graph.pb
    │   ├── model.ckpt.data-00000-of-00001
    │   ├── model.ckpt.index
    │   ├── model.ckpt.meta
    │   ├── pipeline.config
    │   └── saved_model
    ├── large
    │   ├── eval
    │   ├── export
    │   ├── infer
    │   └── train
    └── small
        ├── eval
        ├── export
        ├── infer
        └── train
```
---
#### Numpy Structure:

* First three channels are RGB  
* Afterwhich are annotation masks (one per channel)  
* No of mobile towers= No of channels -3  

----
### Checkpoints

```
/home/user/checkpoint  
|-- faster_rcnn  
|   |-- model.ckpt.data-00000-of-00001  
|   |-- model.ckpt.index  
|   |-- model.ckpt.meta  
|-- mask_rcnn  
|-- ssd  
|-- frcnn_nasnet
|-- frcnn_xception
```
-----
### Models
```
/media/ssd1/face/models  
|-- ssdlite_v2  
|   |-- large  
|   |   |-- small  
|   |   |-- train  
|   |   |   |-- checkpoint  
|   |   |   |-- events.out.tfevents.1526622340.6451ed7d8c29  
|   |   |   |-- graph.pbtxt  
|   |   |   |-- model.ckpt-0.data-00000-of-00001  
|   |   |   |-- model.ckpt-0.index  
|   |   |   |-- model.ckpt-0.meta  
|   |   |   |-- pipeline.config  
|   |   |-- training  
|   |       |-- pipeline.config  
|   |-- small  
|       |-- eval  
|       |   |-- events.out.tfevents.1526537089.6451ed7d8c29  
|       |   |-- pipeline.config  
|       |-- train  
|       |   |-- checkpoint  
|       |   |-- events.out.tfevents.1526536872.6451ed7d8c29  
|       |   |-- graph.pbtxt  
|       |   |-- model.ckpt-18999.data-00000-of-00001  
|       |   |-- model.ckpt-18999.index  
|       |   |-- model.ckpt-18999.meta  
|       |   |-- pipeline.config  
|       |-- training  
|           |-- pipeline.config  

```
----
#### Descriptions:

Each model has a large and a small directory. Small refers to an experimental dataset of just 10 samples.

* `train` : Here all checkpoints and training logs are stored.  
* `eval`: Here all validation logs are stored.  
* `training`: Here the pipeline is stored under `training/pipeline.config`  

----
## Pipeline Code

### Test Images for the inference notebook:

Run `bash object_detection/test_images/get_images.sh`

----

### Dataset Creation (to TF Records)

```
python object_detection/dataset_tools/create_tf_records_faces.py \
--data_dir=/media/ssd1/face/data/WIDER_train/images \
--output_dir=/media/ssd1/face/records \
--text_file_path=/media/ssd1/face/annotations/wider_face_train_bbx_gt.txt \
--mode=train \
--small=${SMALL}
```

---
### Running training

Note :

Have undone the commit caused by [this, commit 93b8168ad](https://github.com/tensorflow/models/commit/93b8168ad9c54f9acf09a161a6dad1dd99b3bfeb)

See the corresponding commit thread for details.

---
#### Large

```
python object_detection/train.py \  
--logtostderr \  
--pipeline_config_path=detection/configs/ssdlite_mobilenet_v2.config \  
--train_dir=/media/ssd1/face/ssdlite_v2/large/train  
```
Clear out any checkpoints under `train_dir` to prevent  

WARNING:root:Variable *XXXX* not found in checkpoint.  

**For multi-use_gpu add these flags:**
--num_clones=${NUM-GPUs} --ps_tasks=${NUM-ps_tasks}

Modify the batch-size accordingly, note that this is not per GPU batch-size.

---
### Running eval
```
python object_detection/eval.py \  
--logtostderr \  
--pipeline_config_path=detection/configs/ssdlite_mobilenet_v2.config \  
--checkpoint_dir=/media/ssd1/face/ssdlite_v2/large/train \  
--eval_dir=/media/ssd1/face/ssdlite_v2/large/eval  
```

Similarly for Small:

```
python object_detection/eval.py \  
--logtostderr \  
--pipeline_config_path=detection/configs/ssdlite_mobilenet_v2_small.config \  
--checkpoint_dir=/media/ssd1/face/ssdlite_v2/small/train \  
--eval_dir=/media/ssd1/face/ssdlite_v2/small/eval  
```

Replace faster_rcnn by mask_rcnn or ssd for the other two models.

---
### Exporting Checkpoint to Protobuf (frozen_inference)

```
python export_inference_graph \  
--input_type "image_tensor" \  
--pipeline_config_path /media/ssd1/face/ssdlite_v2/large/training/pipeline.config \  
--trained_checkpoint_prefix /media/ssd1/face/ssdlite_v2/large/train/{LATEST_CHECKPOINT}\  
--output_directory /media/ssd1/face/ssdlite_v2/large/export/
```

For the latest checkpoint, use

```
python object_detection/export_latest_inference_graph.py \  
--input_type "image_tensor" \
--pipeline_config_path /media/ssd1/face/ssdlite_v2/large/training/pipeline.config \   
--trained_checkpoint_path /media/ssd1/face/ssdlite_v2/large/train \  
--output_directory /media/ssd1/face/ssdlite_v2/large/export  
```

Replace large by small for your needs.


Exporting for Batch Usage:

```
python object_detection/export_latest_inference_graph.py \
--input_type "image_tensor" \labelname_to_imageid
--pipeline_config_path /media/ssd1/face/ssdlite_v2/large/training/pipeline.config \
--trained_checkpoint_path /media/ssd1/face/ssdlite_v2/large/train \
--output_directory /media/ssd1/face/ssdlite_v2/large/export_batch \
--input_shape 5,-1,-1,3
```

To overrride configs (say low proposals):

```
python object_detection/export_latest_inference_graph.py \
--input_type "image_tensor" \
--pipeline_config_path /media/ssd1/face/ssdlite_v2/large/training/pipeline.config \
--trained_checkpoint_path /media/ssd1/face/ssdlite_v2/large/train \
--output_directory /media/ssd1/face/ssdlite_v2/large/export_batch \
--input_shape 5,-1,-1,3
--config_override " \
          model{ \
            faster_rcnn { \
              first_stage_max_proposals: 100 \
              second_stage_post_processing { \
                batch_non_max_suppression { \
                  max_total_detections: 40 \
                } \
              } \
            } \
          }"
```

---
### Running Inference

*Use only 1 GPU for this.*
Added python 3 support.

Refer issue thread [here](https://github.com/tensorflow/models/issues/3903).

```
python object_detection/inference/infer_detections.py \
--input_tfrecord_paths=/media/ssd1/face/records/face_test.record \
--output_tfrecord_path=/media/ssd1/face/ssdlite_v2/large/detections.tfrecord \
--inference_graph=/media/ssd1/face/ssdlite_v2/large/export/frozen_inference_graph.pb
```

Another Example:
```
python object_detection/inference/infer_detections.py \
--input_tfrecord_paths=/media/ssd1/sat_data/face_test.record \
--output_tfrecord_path=/media/ssd1/sat_data_inferences/mask_rcnn/large/detections.tfrecord \
--inference_graph=/media/ssd1/face/ssdlite_v2/large/export_batch_low_proposals/frozen_inference_graph.pb
```

```
python object_detection/inference/run_inference_image.py \
--inference_graph=/media/ssd1/face/ssdlite_v2/large/export/frozen_inference_graph.pb \
--image=/media/ssd1/face/WILDER_test/178.png \
--path_protofile=detection/configs/face.pbtxt \
--output_path=a.png
```
---
### Visualising in-out inference (sorted by IoU), Benchmarking

```
python object_detection/metrics/offline_per_image_eval_map_corloc.py \
--eval_dir=object_detection \
--eval_config_path=object_detection/visualisation_configs/eval.config \
--input_config_path=object_detection/visualisation_configs/faster_rcnn/input_reader.config
```

Involved fixing these issues:

* (None type for groundtruth difficulty flags)[https://stackoverflow.com/questions/47506730/tf-object-detection-api-compute-evaluation-measures-failed/50507864#50507864] :   
  If this were skipped in your TF Records.

* (Byte parsing issues)[https://github.com/tensorflow/models/issues/3252] :   
  If tf_example.features.feature[self.field_name].bytes_list.value returns a byte type instead of string type in metrics/tf_example_parser.StringParser.

Running for other models:

* Evaluation parameters are shared from object_detection/visualisation_configs
* Change input configs.
* Run inference prior to this.

**Visulalisation Config Structure:**
```
object_detection/visualisation_configs
|-- eval.config
|-- faster_rcnn
|   `-- input_reader.config
`-- mask_rcnn
    `-- input_reader.config
```