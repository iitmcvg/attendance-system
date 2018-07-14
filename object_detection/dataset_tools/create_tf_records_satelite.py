r"""Convert the Satellite images into TF Record format

Creates test, train and validation examples, with a split of:

* 0.7 (train)
* 0.2 (validation)
* residual (test)

Example usage:
    python object_detection/dataset_tools/create_tf_records_satellite.py \
        --data_dir=/media/ssd1/cell_data/v5 \
        --output_dir=~/HV/output_satellite/ \
        --small=${SMALL}\
        --prefix='mask'

"""

import hashlib
import io
import logging
import os
import random
import re

import numpy as np
from PIL import Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.core import standard_fields as fields

flags = tf.app.flags

flags.DEFINE_string('data_dir', '/media/ssd1/cell_data/v5/', 'Root directory to raw satellite dataset.')
flags.DEFINE_string('output_dir', '~/HV/output_satellite/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'object_detection/data/satellite.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('small',False,'Create a small dataset for debugging')
flags.DEFINE_string('prefix','','Prefix for the TF records created')

FLAGS = flags.FLAGS

def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def numpy_to_tf_example(example,label_map_dict,image_subdirectory):



    # Read Annotation array
    # Obtain image path

    img_path = os.path.join(image_subdirectory, example+".png")

    annotation_path=os.path.join(image_subdirectory,example+".npy")
    annotation_matrix=np.load(annotation_path)

    if not os.path.exists(img_path):
        # If image doesnot exist
        # Extract image from the numpy array itself
        image = Image.fromarray(annotation_matrix[:,:,:3].astype('uint8'), 'BGR')
        image.save(img_path)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    key = hashlib.sha256(encoded_png).hexdigest()

    # If numpy convention
    width = int(annotation_matrix.shape[1])
    height = int(annotation_matrix.shape[0])

    # If PIL Convention
    #width,height=image.size

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    difficult_obj = []
    masks=[]
    masks_classes=[]

    # Number of towers plus 3
    n_towers=annotation_matrix.shape[2]

    # Iterate over towers
    for i in range(3,n_towers):

        # None of them are difficult objects
        difficult=0
        difficult_obj.append(int(difficult))

        annotate=annotation_matrix[:,:,i]

        non_zero=np.where(annotate!=0)

        xmin = float(np.min(non_zero[1]))
        xmax = float(np.max(non_zero[1]))
        ymin = float(np.min(non_zero[0]))
        ymax = float(np.max(non_zero[0]))
        class_name = "mobile_tower"
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

        # Compute bounding box labels
        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)

        if FLAGS.prefix=="masks":
            masks.append(annotation_matrix[:,:,i])
            masks_classes.append(label_map_dict[class_name])

    # Construct feature dictionary
    feature_dict = {'image/height': dataset_util.int64_feature(height),
     'image/width': dataset_util.int64_feature(width),
     'image/filename': dataset_util.bytes_feature(
         example.encode('utf8')),
     'image/source_id': dataset_util.bytes_feature(
         example.encode('utf8')),
     'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
     'image/encoded': dataset_util.bytes_feature(encoded_png),
     'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
     'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
     'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
     'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
     'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
     'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
     'image/object/class/label': dataset_util.int64_list_feature(classes),
     'image/object/difficult': dataset_util.int64_list_feature(difficult_obj)
    }
    if masks:
        encoded_mask_png_list = []
        for mask in masks:
            img = Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())

        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
        #feature_dict[fields.TfExampleFields.instance_masks] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
        #feature_dict[fields.TfExampleFields.instance_classes] = (dataset_util.int64_list_feature(masks_classes))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_sat_tf_record(output_filename,label_map_dict,data_dir,examples,small=False):

    """Creates a TFRecord file from examples.

Args:
output_filename: Path to where output file is saved.

label_map_dict: The label map dictionary.

data_dir: Directory where data is stored.

examples: Examples to parse and save to tf record."""

    writer = tf.python_io.TFRecordWriter(output_filename)
    # Count Invalid examples (if either image or numpy doesnt exist)
    count=0

    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            print(idx," th Sample")
            logging.info('On image %d of %d', idx, len(examples))
        if small:
            if idx==10:
                break

        numpy_path= os.path.join(data_dir, example + '.npy')
        mask_path = os.path.join(data_dir, example + '.jpg')
        image_path = os.path.join(data_dir, example + '.png')

        if not os.path.exists(numpy_path):
            logging.warning('Could not find %s, ignoring example.', numpy_path)
            count+=1
            continue
        '''
        if not os.path.exists(image_path):
            logging.warning('Could not find %s, ignoring example.', image_path)
            count+=1
            continue
        '''

        try:
            tf_example = numpy_to_tf_example(example,label_map_dict,data_dir)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', numpy_path)

    writer.close()

    # Detect total number of valid examples
    valid_examples=len(examples)-count
    return valid_examples

def extract_examples(examples_path):
    '''
    Extract satellite files for training

    Extracts only those which have an associated .npy file. (Necessary)

    '''
    import glob,os

    examples_list=[]

    for file in glob.glob(examples_path+"/"+"*.npy"):
        examples_list.append(file[:-4])
    return examples_list

def main(_):
    '''
    Run with tf.app.run()
    '''
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Pet dataset.')
    image_dir = data_dir
    annotations_dir = data_dir
    examples_path = data_dir
    examples_list = extract_examples(examples_path)
    small=FLAGS.small
    # Extract names of .jpg files

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    num_val=int(0.2*num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:num_val+num_train]
    test_examples=examples_list[num_train+num_val:]
    logging.info('%d training and %d validation examples and %d test examples.',
               len(train_examples), len(val_examples),len(test_examples))

    if FLAGS.prefix:
        prefix="_"+FLAGS.prefix
    if small:
        train_output_path = os.path.join(FLAGS.output_dir, 'satellite_small_train'+ prefix +'.record')
        val_output_path = os.path.join(FLAGS.output_dir, 'satellite_small_val'+ prefix +'.record')
        test_output_path=os.path.join(FLAGS.output_dir, 'satellite_small_test'+ prefix +'.record')

    else:
        train_output_path = os.path.join(FLAGS.output_dir, 'satellite_train'+ prefix +'.record')
        val_output_path = os.path.join(FLAGS.output_dir, 'satellite_val' + prefix + '.record')
        test_output_path=os.path.join(FLAGS.output_dir, 'satellite_test' + prefix + '.record')

    valid_train_examples=create_sat_tf_record(
      train_output_path,
      label_map_dict,
      data_dir,
      train_examples,small)
    valid_val_examples=create_sat_tf_record(
      val_output_path,
      label_map_dict,
      data_dir,
      val_examples,small)

    valid_test_examples=create_sat_tf_record(
      test_output_path,
      label_map_dict,
      data_dir,
      test_examples,small)

    print("Total feed in data\t", num_examples)
    print("Total Examples for train \t",len(train_examples))
    print("Of which valid \t",valid_train_examples)

    print("\nTotal Examples for validation \t",len(val_examples))
    print("Of which valid \t",valid_val_examples)

    print("\nTotal Examples for test \t",len(test_examples))
    print("Of which valid \t",valid_test_examples)

    print("Successfully converted.")

if __name__ == '__main__':
    tf.app.run()
