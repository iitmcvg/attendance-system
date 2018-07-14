r"""Convert the face images into TF Record format

Example usage:
python object_detection/dataset_tools/create_tf_records_faces.py \
--data_dir=/media/ssd1/face/data/WIDER_train/images \
--output_dir=/media/ssd1/face/records \
--text_file_path=/media/ssd1/face/annotations/wider_face_train_bbx_gt.txt \
--mode=train \
--small=${SMALL}


"""

import hashlib
import io
import logging
import os
import random
import re
import glob

import numpy as np
from PIL import Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.core import standard_fields as fields

flags = tf.app.flags

flags.DEFINE_string('data_dir', '/media/ssd1/face/data/WIDER_train/images', 'Root directory to raw face dataset.')
flags.DEFINE_string('output_dir', '/media/ssd1/face/records', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'detection/configs/face.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('small',False,'Create a small dataset for debugging')
flags.DEFINE_string('text_file_path', '/media/ssd1/face/annotations/wider_face_train_bbx_gt.txt',
                    'Path to label map proto')
flags.DEFINE_string('mode','train','mode to use: train,test or val')

FLAGS = flags.FLAGS

def text_instance_to_tf_example(filename,subarray,label_map_dict,image_subdirectory):
    '''
    filename: (example) 0--Parade/0_Parade_marchingband_1_799.jpg

    subarray: (example) 

    78 221 7 8 2 0 0 0 0 0 
    78 238 14 17 2 0 0 0 0 0 
    113 212 11 15 2 0 0 0 0 0 
    134 260 15 15 2 0 0 0 0 0 
    163 250 14 17 2 0 0 0 0 0 
    201 218 10 12 2 0 0 0 0 0 
    182 266 15 17 2 0 0 0 0 0 
    245 279 18 15 2 0 0 0 0 0 

    '''
    def _get_bounds(x1,y1,w,h):
        xmin=x1
        ymin=y1
        xmax=xmin+w
        ymax=ymin+h
        return (xmin,ymin,xmax,ymax)

    def _check(xmin,ymin,xmax,ymax):
        return xmin<xmax and ymin<ymax

    # Obtain image path
    img_path = os.path.join(image_subdirectory,filename)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)

    image = Image.open(encoded_png_io)
    key = hashlib.sha256(encoded_png).hexdigest()

    # If numpy convention
    width ,height = image.size

    # If PIL Convention
    #width,height=image.size
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    difficult_obj = []

    # Iterate over towers
    for i in range(len(subarray)):

        line=subarray[i].split(" ")
        line=[int(j) for j in line if j]
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose=line
        if len(line)!=10:
            print(line)
            break
        xmin,ymin,xmax,ymax=_get_bounds(x1,y1,w,h)
        if not _check(xmin,ymin,xmax,ymax):
            print(xmin,ymin,xmax,ymax)
            break

        # None of them are difficult objects
        difficult=False
        if occlusion==2 or blur==2:
            difficult=True
        difficult_obj.append(int(difficult))
        class_name = "face"
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

        # Compute bounding box labels
        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)

    # Construct feature dictionary
    feature_dict = {'image/height': dataset_util.int64_feature(height),
     'image/width': dataset_util.int64_feature(width),
     'image/filename': dataset_util.bytes_feature(
         filename.encode('utf8')),
     'image/source_id': dataset_util.bytes_feature(
         filename.encode('utf8')),
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

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_face_tf_record(output_filename,label_map_dict,data_dir,text_file,small=False):
    """
    Creates a TFRecord file from examples.

    Args:
    output_filename: Path to where output file is saved.

    label_map_dict: The label map dictionary.

    data_dir: Directory where data is stored.

    text_file: Annotation text file to use.
    """
    valid_examples=0

    files=np.array(glob.glob(os.path.join(data_dir,"*/*.jpg")))
    files=["/".join(file.split("/")[-2:]) for file in files]
    tf.logging.debug("files examples {}".format(files[:3]))
    with open(text_file,'r') as f:
        data=f.read().splitlines()
        data=np.array(data)

    writer = tf.python_io.TFRecordWriter(output_filename)

    indices=np.where(np.isin(data,files))[0]
    tf.logging.debug('indices {}'.format(indices))
    #tf.logging.debug("INDICES {}".format(indices[:10]))
    for i in range(len(indices)):
        indice=indices[i]

        if i==len(indices)-1:
            indice_next=-1
        else:
            indice_next=indices[i+1]

        if small:
            if i==9:
                break
        tf.logging.debug("indice {} indice_next {}".format(indice,indice_next))
        subarray=data[indice+2:indice_next]
        image_path=data[indice]
        image_subdirectory=data_dir

        try:
            tf_example = text_instance_to_tf_example(image_path,subarray,label_map_dict,image_subdirectory)
            writer.write(tf_example.SerializeToString())
            valid_examples+=1
        except Exception as e:
            raise e
        '''
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', image_path)
        '''
    writer.close()
    return valid_examples

def main(_):
    '''
    Run with tf.app.run()
    '''
    tf.logging.set_verbosity(tf.logging.DEBUG)

    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    logging.info('Reading from face dataset.')
    small=FLAGS.small

    if small:
        output_path = os.path.join(FLAGS.output_dir, 'face_small_'+ FLAGS.mode +'.record')
    else:
        output_path = os.path.join(FLAGS.output_dir, 'face_'+ FLAGS.mode +'.record')

    valid_examples=create_face_tf_record(
      output_path,
      label_map_dict,
      data_dir,
      FLAGS.text_file_path\
      ,small)
    
    print("Total feed in data\t", len(glob.glob(os.path.join(data_dir,"*/*.jpg"))))
    print("Of which valid \t",valid_examples)

    print("Successfully converted.")

if __name__ == '__main__':
    tf.app.run()
