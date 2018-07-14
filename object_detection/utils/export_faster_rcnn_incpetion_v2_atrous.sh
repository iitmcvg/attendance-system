
# From tensorflow/models/research/
pbPATH="object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28"


EXPORT_DIR="object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28_tf_14"
mkdir EXPORT_DIR

echo which python
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${pbPATH}/pipeline.config  \
    --trained_checkpoint_prefix ${pbPATH}/model.ckpt \
    --output_directory ${EXPORT_DIR}
