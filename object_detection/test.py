import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


inference_path='/media/ssd1/sat_data_inferences/faster_rcnn/large/detections.tfrecord'
test_path='/media/ssd1/sat_data/satellite_test.record'

def read_input_data(data_path='/media/ssd1/sat_data/satellite_test.record'):
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        feature = {'image/height': tf.FixedLenFeature([], tf.int64),
                   'image/width': tf.FixedLenFeature([], tf.int64),
                   'image/filename': tf.FixedLenFeature([], tf.string),
                   'image/source_id': tf.FixedLenFeature([], tf.string),
                   'image/key/sha256': tf.FixedLenFeature([], tf.string),
                   'image/encoded': tf.FixedLenFeature([], tf.string),
                   'image/format': tf.FixedLenFeature([], tf.string),
                   'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                   'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                   'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                   'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                   'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
                   'image/object/class/label': tf.VarLenFeature(dtype=tf.float32)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        input_data_dict={}

        tf.train.start_queue_runners(sess)

        try:
            while True:
                # Decode the record read by the reader
                features = tf.parse_single_example(serialized_example, features=feature)

                print(features.items())
                for name, tensor in features.items():
                    print('{}: {}'.format(name, tensor.eval()))
                    break

                break

                # Convert the image data from string back to the numbers
                image = tf.decode_raw(features['image/encoded'], tf.float32)

                # Read filename
                filename=features['image/filename']

                # Read height and width
                height=tf.decode_raw(features['image/height'],tf.int64)
                width=features['image/width']
                print(height)

                # Reshape image data into the original shape
                image = tf.reshape(image, [width, height, 3])

                # Cast label data into int32
                label = tf.cast(features['image/object/class/label'], tf.int32)

                text_label=features['image/object/class/text']

                # Read annotation
                xmin=tf.cast(features['image/object/bbox/xmin'],tf.int64)
                xmax=tf.cast(features['image/object/bbox/xmax'],tf.int64)
                ymin=tf.cast(features['image/object/bbox/ymin'],tf.int64)
                ymax=tf.cast(features['image/object/bbox/ymax'],tf.int64)

                annotation={'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax}

                # Add to output dictionary

                input_data_dict[filename]={'image':image,'label':label,'text_label':label,'annotation':annotation}

                print(input_data_dict)

                break

        except Exception as e:
            print (e.message, e.args)
            print("Done reading in dictionary")

        finally:
            pass

read_input_data()
