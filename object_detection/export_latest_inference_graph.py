'''
Exports latest inference graph

Usage:

Example Usage:
--------------
python export_latest_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_path path/to/model \
    --output_directory path/to/exported_model_directory


Will reinstate directory if exist.

Args:

trained_checkpoint_prefix: path to folder containing all ckpts


The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
 - graph.pbtxt
 - model.ckpt.data-00000-of-00001
 - model.ckpt.info
 - model.ckpt.meta
 - frozen_inference_graph.pb
 + saved_model (a directory)

Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage (in which we change the second stage post-processing score
threshold to be 0.5):

python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_path [folder containing]path/to/model.ckpt \
    --output_directory path/to/exported_model_directory \
    --config_override " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"

'''

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

import glob,os
import shutil
import subprocess
import re

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('input_shape', None,
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_path', None,
                    'Folder path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('trained_checkpoint_path')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS

'''

if FLAGS.input_shape and FLAGS.config_override:

    cmd="python object_detection/export_inference_graph.py --input_type image_tensor" \
    +" --pipeline_config_path " + FLAGS.pipeline_config_path\
    +" --trained_checkpoint_prefix "+ checkpoint_path\
    +" --output_directory "+ FLAGS.output_directory\
    +" --input_shape "+ FLAGS.input_shape\
    +" --config_override "+ FLAGS.config_override

elif FLAGS.input_shape:

    cmd="python object_detection/export_inference_graph.py --input_type image_tensor" \
    +" --pipeline_config_path " + FLAGS.pipeline_config_path\
    +" --trained_checkpoint_prefix "+ checkpoint_path\
    +" --output_directory "+ FLAGS.output_directory\
    +" --input_shape "+ FLAGS.input_shape

elif FLAGS.config_override:

    cmd="python object_detection/export_inference_graph.py --input_type image_tensor" \
    +" --pipeline_config_path " + FLAGS.pipeline_config_path\
    +" --trained_checkpoint_prefix "+ checkpoint_path\
    +" --output_directory "+ FLAGS.output_directory\
    +" --config_override "+ FLAGS.config_override

    print(cmd)

else:
    cmd="python object_detection/export_inference_graph.py --input_type image_tensor" \
    +" --pipeline_config_path " + FLAGS.pipeline_config_path\
    +" --trained_checkpoint_prefix "+ checkpoint_path\
    +" --output_directory "+ FLAGS.output_directory

p = subprocess.Popen(cmd, shell=True)

print(p.communicate())

if not p.communicate()[0]:
    print("Exporting inference graph done successfully.")

'''

def main(_):
    # Reinstate export path if needed

    if os.listdir(FLAGS.output_directory):
        shutil.rmtree(FLAGS.output_directory)
        os.mkdir(FLAGS.output_directory)

    # Find latest checkpoint
    checkpoint_path=sorted(glob.glob(FLAGS.trained_checkpoint_path+"/*.ckpt*.data*"))[-1]
    checkpoint_path=re.match("\S*.ckpt-([0-9]*)",checkpoint_path).group()

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
        text_format.Merge(FLAGS.config_override, pipeline_config)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(FLAGS.input_type, pipeline_config,
                                  checkpoint_path,
                                  FLAGS.output_directory, input_shape)


if __name__ == '__main__':
  tf.app.run()
