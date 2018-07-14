import time
import numpy as np
from PIL import Image
import tensorflow as tf
import os,glob
import itertools
import io
import scipy

import json


from object_detection.inference import detection_inference
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import label_map_util

def fetch_array(coordinate,image_path):
        '''
        Read image array from path
        '''
        path=os.path.join(image_path,str(coordinate[0]),str(coordinate[1])+".png")
        return scipy.ndimage.imread(path,mode="RGB").astype(np.uint8)

def build_inference_graph(image_tensor, graph_content,batch_size=1):

        """Loads the inference graph and connects it to the input image.

        Args:
        image_tensor: The input image. uint8 tensor, shape=[1, None, None, 3]
        inference_graph_path: Path to the inference graph with embedded weights

        Returns:
        detected_boxes_tensor: Detected boxes. Float tensor,
            shape=[num_detections, 4]
        detected_scores_tensor: Detected scores. Float tensor,
            shape=[num_detections]
        detected_labels_tensor: Detected labels. Int64 tensor,
            shape=[num_detections]
        """

        graph_def = tf.GraphDef()
        graph_def.MergeFromString(graph_content)

        tf.import_graph_def(
          graph_def, name='', input_map={'image_tensor': image_tensor})

        g = tf.get_default_graph()

        if batch_size==1:
            num_detections_tensor = tf.squeeze(
              g.get_tensor_by_name('num_detections:0'), 0)
        else:
            num_detections_tensor = g.get_tensor_by_name('num_detections:0')
            tf.logging.debug("NUM DETECTIONS TENSOR.{}".format(num_detections_tensor.shape))

        num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)
        
        if batch_size==1:
            detected_boxes_tensor = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), 0)
            detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]
        else:
            detected_boxes_tensor = g.get_tensor_by_name('detection_boxes:0')

        if batch_size==1:
            detected_scores_tensor = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), 0)
            detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]
        else:
            detected_scores_tensor = g.get_tensor_by_name('detection_scores:0')

        if batch_size==1:
            detected_labels_tensor = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), 0)
            detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)
            detected_labels_tensor = detected_labels_tensor[:num_detections_tensor]
        else:
            detected_labels_tensor = g.get_tensor_by_name('detection_classes:0')
            detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)

        return num_detections_tensor,detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor


def _read_patch(start,path):
    '''
    Returns image tensor I. [Numpy]
    Shape 1024*1024

    Args:
    - path: Path for the files stored as /zoom/x/y
    - start: Start coordinate for the patch, each of size 256*256. A Numpy array

    I[:256,:256]=image(start)
    I[256:512,:256]=image(start+(1,0))
    I[512:512,:256]=image(start+(1,0))
    I[256:512,:256]=image(start+(1,0))

    I[256:512,:256]=image(start+(1,0))
    I[256:512,:256]=image(start+(1,0))
    I[256:512,:256]=image( start+(1,0))
    I[256:512,:256]=image(start+(1,0))

    '''
    start=np.array(start)
    hor=np.array([1,0])
    ver=np.array([0,1])
    covered=[]
    for i in range(4):
        start1=start.copy()
        for j in range(4):
            if j==0:
                # Intialise I1
                try:
                    I1=fetch_array(start1,path)
                    covered.append(str(start[0])+"/"+str(start[1]))
                except:
                    I1=np.zeros((256,256,3))
                    covered.append(str(start[0])+"/"+str(start[1])+"-Missed")
            else:
                try:
                    I1=np.hstack((I1,fetch_array(start1,path)))
                    covered.append(str(start[0])+"/"+str(start[1]))
                except Exception as e:
                    I1=np.hstack((I1,np.zeros((256,256,3))))
                    covered.append(str(start[0])+"/"+str(start[1])+"-Missed")
            start1+=hor
        if i==0:
            I=I1
        else:
            I=np.vstack((I,I1))
        start=start+ver

    return (I,covered)


path="/media/ssd1/data/1040010010608200_jpg.tif_tiles/19/"
start=(121390,211372)
path_protofile= "/media/ssd1/sat_data/satellite.pbtxt"
inference_graph_path="/media/ssd1/sat_data_models/mask_rcnn/large/export_low_proposals/frozen_inference_graph.pb"
batch_size=1
I,_=_read_patch(start,path)


# Enable Verbose Logging
tf.logging.set_verbosity(tf.logging.INFO)


# Load category map
'''
A category index, which is a dictionary that maps integer ids to dicts
containing categories, e.g.
{1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}
'''

category_index_from_labelmap=label_map_util.create_category_index_from_labelmap(path_protofile)

with tf.Session() as sess:

    # Obtain image tensor
    I=I[np.newaxis,:]
    I=np.repeat(I,batch_size,axis=0)
    image_tensor=tf.constant(I)

    # Run graph
    tf.logging.info('Reading graph and building model...')

    with tf.gfile.Open(inference_graph_path, 'rb') as graph_def_file:
        graph_content = graph_def_file.read()

    tf.logging.info('Graph loaded')
    (v,detected_boxes_tensor, detected_scores_tensor,\
    detected_labels_tensor) = build_inference_graph(image_tensor, graph_content,batch_size=batch_size)
    tf.logging.info('Getting detections')
    # Get detections
    (detected_boxes, detected_scores,\
    detected_labels)=sess.run([detected_boxes_tensor, detected_scores_tensor,\
    detected_labels_tensor])
    tf.logging.info("ran session")

    # Detected boxes of form: [ymins,xmins,ymax,xmax]

    input_image=sess.run(image_tensor)
    print(input_image)
    input_image=(input_image[0])

    # Draw bounding boxes
   
    ii=np.where(detected_scores>0.4)
    for i in range(len(detected_scores[ii])):
        ymin=detected_boxes[i][0]
        xmin=detected_boxes[i][1]
        ymax=detected_boxes[i][2]
        xmax=detected_boxes[i][3]

        category=category_index_from_labelmap[detected_labels[i]]['name']

        vis_utils.draw_bounding_box_on_image_array(input_image,xmin=xmin,ymin=ymin,\
        xmax=xmax, ymax=ymax,display_str_list=(category) ,color='MediumPurple')

    vis_utils.save_image_array_as_png(input_image,'e.png')
