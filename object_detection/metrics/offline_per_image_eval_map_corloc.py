# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Evaluation executable for detection data.

This executable evaluates precomputed detections produced by a detection
model and writes the evaluation results into csv file metrics.csv, stored
in the directory, specified by --eval_dir.

The evaluation metrics set is supplied in object_detection.protos.EvalConfig
in metrics_set field.
Currently two set of metrics are supported:
- pascal_voc_metrics: standard PASCAL VOC 2007 metric
- open_images_detection_metrics: Open Image V2 metric
All other field of object_detection.protos.EvalConfig are ignored.

Example usage:
    ./compute_metrics \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=path/to/evaluation/configuration/file \
        --input_config_path=path/to/input/configuration/file
"""
import csv
import os
import re
import tensorflow as tf
import numpy as np

from object_detection import evaluator
from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import config_util
from object_detection.utils import label_map_util

from object_detection.utils import per_image_custom_evaluation as per_image_evaluation

import pandas as pd

flags = tf.app.flags
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('eval_dir', None, 'Directory to write eval summaries to.')
flags.DEFINE_string('eval_config_path', None,
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', None,
                    'Path to an eval_pb2.InputConfig config file.')
flags.DEFINE_float('score_threshold', 0.5,
                    'score threshold to use for precision')

FLAGS = flags.FLAGS


def _generate_sharded_filenames(filename):
  m = re.search(r'@(\d{1,})', filename)
  if m:
    num_shards = int(m.group(1))
    return [
        re.sub(r'@(\d{1,})', '-%.5d-of-%.5d' % (i, num_shards), filename)
        for i in range(num_shards)
    ]
  else:
    return [filename]


def _generate_filenames(filenames):
  result = []
  for filename in filenames:
    result += _generate_sharded_filenames(filename)
  return result


def read_data_and_evaluate(input_config, eval_config,score_threshold=0):
  """Reads pre-computed object detections and groundtruth from tf_record.

  Args:
    input_config: input config proto of type
      object_detection.protos.InputReader.
    eval_config: evaluation config proto of type
      object_detection.protos.EvalConfig.

  Returns:
    Evaluated detections metrics.


  Format:

  filename: metrics

  Raises:
    ValueError: if input_reader type is not supported or metric type is unknown.
  """
  if input_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    input_paths = input_config.tf_record_input_reader.input_path
    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes)

    '''
    filename: metric list
    '''
    metrics_image_map={}

    # Setup metrics map

    metrics_image_map['filename']=[]
    metrics_image_map['is_atleast_one_mobile_tower_detected']=[]
    metrics_image_map['result_scores_class_1']=[]
    metrics_image_map['result_tp_fp_labels_class_1']=[]
    metrics_image_map['recall']=[]
    metrics_image_map['precision (greater than threshold)']=[]

    skipped_images = 0
    processed_images = 0

    per_image_eval=per_image_evaluation.PerImageEvaluation(1,matching_iou_threshold=0.2)

    for input_path in _generate_filenames(input_paths):
      tf.logging.info('Processing file: {0}'.format(input_path))

      record_iterator = tf.python_io.tf_record_iterator(path=input_path)
      data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

      for string_record in record_iterator:
        # Image metric set

        tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 100,
                               processed_images)
        processed_images += 1

        example = tf.train.Example()
        example.ParseFromString(string_record)

        decoded_dict = data_parser.parse(example)

        if decoded_dict:
          detected_boxes=decoded_dict[standard_fields.DetectionResultFields.detection_boxes]
          detected_scores=decoded_dict[standard_fields.DetectionResultFields.detection_scores]
          detected_class_labels=decoded_dict[standard_fields.DetectionResultFields.detection_classes]-1

          groundtruth_boxes=decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
          groundtruth_class_labels=decoded_dict[standard_fields.InputDataFields.groundtruth_classes]-1
          groundtruth_is_difficult_list=decoded_dict[standard_fields.InputDataFields.groundtruth_difficult]
          groundtruth_is_group_of_list=np.zeros(len(groundtruth_boxes),dtype=np.int64)!=0

          scores, tp_fp_labels, is_class_correctly_detected_in_image = \
          per_image_eval.compute_object_detection_metrics(detected_boxes=detected_boxes,\
           detected_scores=detected_scores,\
            detected_class_labels=detected_class_labels,\
          groundtruth_boxes=groundtruth_boxes, \
          groundtruth_class_labels=groundtruth_class_labels,\
           groundtruth_is_difficult_list=groundtruth_is_difficult_list,\
          groundtruth_is_group_of_list=groundtruth_is_group_of_list)

          '''
          Returns:
                scores: A list of C float numpy arrays. Each numpy array is of
                    shape [K, 1], representing K scores detected with object class
                    label c
                tp_fp_labels: A list of C boolean numpy arrays. Each numpy array
                    is of shape [K, 1], representing K True/False positive label of
                    object instances detected with class label c
                is_class_correctly_detected_in_image: a numpy integer array of
                    shape [C, 1], indicating whether the correponding class has a least
                    one instance being correctly detected in the image

          '''

          indices=np.where(scores[0]>score_threshold)

          metrics_image_map['filename'].append(decoded_dict[standard_fields.DetectionResultFields.key])
          metrics_image_map['is_atleast_one_mobile_tower_detected'].append(is_class_correctly_detected_in_image[0])
          metrics_image_map['result_scores_class_1'].append((" - ").join([str(i) for i in scores[0]]))
          metrics_image_map['result_tp_fp_labels_class_1'].append((" - ").join([str(i) for i in tp_fp_labels[0]]))

          metrics_image_map['recall'].append(np.sum(tp_fp_labels[0])/len(groundtruth_class_labels))

          metrics_image_map['precision (greater than threshold)'].append(np.sum(tp_fp_labels[0][indices])/len(tp_fp_labels[0][indices]))

        else:
          skipped_images += 1
          tf.logging.info('Skipped images: {0}'.format(skipped_images))

    return metrics_image_map

  raise ValueError('Unsupported input_reader_config.')


def write_metrics(metrics_image_map, output_dir):
  """Write metrics to the output directory.

  Args:
    metrics: A dictionary containing metric names and values.
    output_dir: Directory to write metrics to.
  """
  tf.logging.info('Writing metrics.')

  df=pd.DataFrame.from_dict(metrics_image_map)
  file_name=os.path.join(output_dir, 'metrics.csv')
  df.to_csv(file_name, sep='\t', encoding='utf-8')


def main(argv):
  del argv
  required_flags = ['input_config_path', 'eval_config_path', 'eval_dir']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  configs = config_util.get_configs_from_multiple_files(
      eval_input_config_path=FLAGS.input_config_path,
      eval_config_path=FLAGS.eval_config_path)

  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']

  metrics_image_map = read_data_and_evaluate(input_config, eval_config,FLAGS.score_threshold)

  # Save metrics
  write_metrics(metrics_image_map, FLAGS.eval_dir)


if __name__ == '__main__':
  tf.app.run(main)
