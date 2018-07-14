'''
Visualise outputs of the inference script.

For :

* IoU
* Image boxes


Note:

all coordinates (bounding boxes) are normalised.
'''

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
import re

import constants

# Fields for the TF Records
from object_detection.core import standard_fields

def read_and_decode_input(filename_queue):
    '''
    Decode input TF Records
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
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
               'image/object/class/label': tf.VarLenFeature(dtype=tf.int64)}
    features = tf.parse_single_example(serialized_example,feature)

    # Read encoded image
    encoded_image = features['image/encoded']
    image = tf.image.decode_image(encoded_image, channels=3)

    # Read labels and filename
    label = tf.cast(features['image/object/class/label'], tf.int32)
    text_label=features['image/object/class/text']
    filename=features['image/filename']

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    # Read bounding boxes
    xmins=tf.cast(features['image/object/bbox/xmin'],tf.int64)
    xmaxs=tf.cast(features['image/object/bbox/xmax'],tf.int64)
    ymins=tf.cast(features['image/object/bbox/ymin'],tf.int64)
    ymaxs=tf.cast(features['image/object/bbox/ymax'],tf.int64)

    return image, filename,label,text_label, height, width, xmins,xmaxs,ymins,ymaxs

def read_and_decode_output(filename_queue):
    '''
    Decode output TF Records
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
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
               'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
               standard_fields.TfExampleFields.detection_score:tf.VarLenFeature(dtype=tf.int64),
               standard_fields.TfExampleFields.detection_bbox_ymin:tf.VarLenFeature(dtype=tf.int64),
               standard_fields.TfExampleFields.detection_bbox_xmin:tf.VarLenFeature(dtype=tf.int64),
               standard_fields.TfExampleFields.detection_bbox_ymax:tf.VarLenFeature(dtype=tf.int64),
               standard_fields.TfExampleFields.detection_bbox_xmax:tf.VarLenFeature(dtype=tf.int64),
               standard_fields.TfExampleFields.detection_class_label:tf.VarLenFeature(dtype=tf.int64)}

    features = tf.parse_single_example(serialized_example,feature)

    # Read encoded image
    encoded_image = features['image/encoded']
    filename=features['image/filename']
    image = tf.image.decode_image(encoded_image, channels=3)

    # Read labels and filename
    truth_labels = tf.cast(features['image/object/class/label'], tf.int32)
    truth_text_labels=features['image/object/class/text']


    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    # Read ground truth bounding boxes
    truth_xmins=tf.cast(features['image/object/bbox/xmin'],tf.int64)
    truth_xmaxs=tf.cast(features['image/object/bbox/xmax'],tf.int64)
    truth_ymins=tf.cast(features['image/object/bbox/ymin'],tf.int64)
    truth_ymaxs=tf.cast(features['image/object/bbox/ymax'],tf.int64)

    # Read detection scores
    detection_score=features[standard_fields.TfExampleFields.detection_score]

    # Read detected bounding boxes
    detected_xmins=tf.cast(features[standard_fields.TfExampleFields.detection_bbox_xmin],tf.int64)
    detected_xmaxs=tf.cast(features[standard_fields.TfExampleFields.detection_bbox_xmax],tf.int64)
    detected_ymins=tf.cast(features[standard_fields.TfExampleFields.detection_bbox_ymin],tf.int64)
    detected_ymaxs=tf.cast(features[standard_fields.TfExampleFields.detection_bbox_ymax],tf.int64)

    # Read detected class labels
    detected_labels=features[standard_fields.TfExampleFields.detection_class_label]

    return [image, filename,truth_labels,truth_text_labels,\
     truth_xmins,truth_xmaxs,truth_ymins,truth_ymaxs,\
     detection_score,detected_labels,\
     detected_xmins,detected_xmaxs,detected_ymins,detected_ymaxs]


def get_all_input_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([ FILE ])
        image, filename, labels,text_labels, height, width, xmins,xmaxs,ymins,ymaxs = read_and_decode_output(filename_queue)

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(4):
            example ,filename_,width_,height_= sess.run([image,filename,width,height])
            filename_=str(filename_,'utf-8')
            filename_=filename_.split("/")[-1]
            print((filename_))
            img = Image.fromarray(example, 'RGB')
            img.save( "output_xxx/" + filename_ + '.jpg')

        coord.request_stop()
        coord.join(threads)

def calculate_metrics


def get_all_output_records(FILE):
    '''
    returns a list of dictionart of outputs

    '''

    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([ FILE ])

        # Read all tensors
        Tensor_List= read_and_decode_output(filename_queue)

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        output_list=[]

        try:
            while True:
                image, filename,truth_labels,truth_text_labels,\
                truth_xmins,truth_xmaxs,truth_ymins,truth_ymaxs,\
                detection_score,detected_labels,\
                detected_xmins,detected_xmaxs,detected_ymins,detected_ymaxs= sess.run(Tensor_List)

                filename=str(filename,'utf-8')
                filename=filename.split("/")[-1]

                OUTPUT_DATA={'image':image,\
                            'filename':filename,\
                            'truth_labels':truth_labels,\
                            'truth_annotation':[truth_xmins,truth_ymins,truth_xmax,truth_ymax],\
                            'detection_score':detection_score,\
                            'detected_labels':detected_labels,\
                            'detected_annotation':}

                output_list.append(OUTPUT_DATA)





        except Exception as e:
            print e.message
            print("Finished reading file")

        coord.request_stop()
        coord.join(threads)

get_all_input_records('/media/ssd1/sat_data/satellite_test.record')
