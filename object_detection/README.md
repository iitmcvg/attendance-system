# README

_Task Objective:_ To detect mobile towers in satellite data.

_Approach :_ Treated as small scale object detection problem Statement

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
/media/ssd1/sat_data/  
|-- satellite.pbtxt  
|-- satellite_small_train.record  
|-- satellite_small_test.record
|-- satellite_small_val.record  
|-- satellite_train.record
|-- satellite_test.record  
|-- satellite_val.record  
```

Tree
```
/media/ssd1/cell_data/v5/  
~~  
|-- 685.npy  
|-- 685.png  
|-- 686.jpg  
~~
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
/media/ssd1/sat_data_models/  
|-- faster_rcnn  
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
|-- mask_rcnn  
|-- rfcn  
|-- ssd  
|-- frcnn_nasnet
|-- frcnn_xception
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
python object_detection/dataset_tools/create_tf_records_satelite.py \  
--data_dir=/media/ssd1/cell_data/v5 \  
--output_dir=/media/ssd1/sat_data/ \  
--small=${SMALL}
--
```
---
### Running training

Note :

Have undone the commit caused by [this, commit 93b8168ad](https://github.com/tensorflow/models/commit/93b8168ad9c54f9acf09a161a6dad1dd99b3bfeb)

See the corresponding commit thread for details.

---
#### Small

```
python object_detection/train.py \  
--logtostderr \  
--pipeline_config_path=/media/ssd1/sat_data_models/faster_rcnn/small/training/pipeline.config \  
--train_dir=/media/ssd1/sat_data_models/faster_rcnn/small/train  
```
---
#### Large

```
python object_detection/train.py \  
--logtostderr \  
--pipeline_config_path=/media/ssd1/sat_data_models/faster_rcnn/large/training/pipeline.config \  
--train_dir=/media/ssd1/sat_data_models/faster_rcnn/large/train  
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
--pipeline_config_path=/media/ssd1/sat_data_models/faster_rcnn/large/training/pipeline.config \  
--checkpoint_dir=/media/ssd1/sat_data_models/faster_rcnn/large/train \  
--eval_dir=/media/ssd1/sat_data_models/faster_rcnn/large/eval  
```
Similar for Small.

Replace faster_rcnn by mask_rcnn or ssd for the other two models.

---
### Exporting Checkpoint to Protobuf (frozen_inference)

```
python export_inference_graph \  
--input_type "image_tensor" \  
--pipeline_config_path /media/ssd1/sat_data_models/faster_rcnn/large/training/pipeline.config \  
--trained_checkpoint_prefix /media/ssd1/sat_data_models/faster_rcnn/large/train/{LATEST_CHECKPOINT}\  
--output_directory /media/ssd1/sat_data_models/faster_rcnn/large/export/
```

For the latest checkpoint, use

```
python object_detection/export_latest_inference_graph.py \  
--input_type "image_tensor" \
--pipeline_config_path /media/ssd1/sat_data_models/faster_rcnn/large/training/pipeline.config \   
--trained_checkpoint_path /media/ssd1/sat_data_models/faster_rcnn/large/train \  
--output_directory /media/ssd1/sat_data_models/faster_rcnn/large/export  
```

Replace large by small for your needs.


Exporting for Batch Usage:

```
python object_detection/export_latest_inference_graph.py \
--input_type "image_tensor" \labelname_to_imageid
--pipeline_config_path /media/ssd1/sat_data_models/faster_rcnn/large/training/pipeline.config \
--trained_checkpoint_path /media/ssd1/sat_data_models/faster_rcnn/large/train \
--output_directory /media/ssd1/sat_data_models/faster_rcnn/large/export_batch \
--input_shape 5,-1,-1,3
```

To overrride configs (say low proposals):

```
python object_detection/export_latest_inference_graph.py \
--input_type "image_tensor" \
--pipeline_config_path /media/ssd1/sat_data_models/faster_rcnn/large/training/pipeline.config \
--trained_checkpoint_path /media/ssd1/sat_data_models/faster_rcnn/large/train \
--output_directory /media/ssd1/sat_data_models/faster_rcnn/large/export_batch_low_proposals \
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
--input_tfrecord_paths=/media/ssd1/sat_data/satellite_test.record \
--output_tfrecord_path=/media/ssd1/sat_data_inferences/faster_rcnn/large/detections.tfrecord \
--inference_graph=/media/ssd1/sat_data_models/faster_rcnn/large/export/frozen_inference_graph.pb
```

Another Example:
```
python object_detection/inference/infer_detections.py \
--input_tfrecord_paths=/media/ssd1/sat_data/satellite_test.record \
--output_tfrecord_path=/media/ssd1/sat_data_inferences/mask_rcnn/large/detections.tfrecord \
--inference_graph=/media/ssd1/sat_data_models/mask_rcnn/large/export_batch_low_proposals/frozen_inference_graph.pb
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
----

### Running Tile Parser

```

python object_detection/inference/data_infer_batch.py \
--input_path="/media/ssd1/data/1040010018255600_jpg.tif_tiles/19/" \
--inference_graph="/media/ssd1/sat_data_models/faster_rcnn/large/export/frozen_inference_graph.pb" \
--output_path="/media/ssd1/tile_output/" \
--shard_path="/media/ssd1/tile_output/" \
--batch_size=1 \
--shard=0 \
--vis_path="/media/ssd1/tile_output/"

```

Runs a `tf.data.Dataset` based pipeline for tiling images.
