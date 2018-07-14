"""
Tests for data_infer_batch.py.

Usage:

python object_detection/inference/cell_images_run.py \
--input_path="/media/ssd1/cell_images/" \
--inference_graph="/media/ssd1/sat_data_models/faster_rcnn/large/export/frozen_inference_graph.pb" \
--batch_size=1 \
--test_size=10 \
--vis_path="/media/ssd1/cell_images_vis" \
--test True


"""

import os

try:
    import StringIO
except ImportError:
    import io as StringIO

# from cStringIO import StringIO
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import os,glob
import itertools
import io
import scipy

import json

# String Flags
tf.flags.DEFINE_string('input_path', None,
                       'Tiled Satellite directory')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights and correct input batching.')
# Integer Flags
tf.flags.DEFINE_integer('batch_size', 1,
                       'Batch size for image runs per graph')

tf.flags.DEFINE_float('threshold', 0.1,
                       'Threshold to use for json')
tf.flags.DEFINE_float('vis_threshold', 0.9,
                       'Threshold to use for visualisation')

tf.flags.DEFINE_string('vis_path', '',
                       'Visualisation path with detections')
tf.flags.DEFINE_boolean('test',False,
                        'Test or not')

FLAGS = tf.flags.FLAGS

import object_detection.utils.visualization_utils as vis_utils


class CellImagesRun(object):
    '''
    Run batchwise on cell images, visualise some
    '''

    def __init__(self,path,batch_size=5):
        self.path=path
        self.batch_size=batch_size
        self.map=glob.glob(os.path.join(self.path,'*.png'))
        self.map=[x.split("/")[-1] for x in self.map ]

        '''
        of form [xxx.png,yyyy.png]
        '''

    def fetch_array(self,image_id):
        '''
        Read image array from path
        '''
        path=os.path.join(self.path,image_id)
        try:
            return (True,scipy.ndimage.imread(path,mode="RGB"))
        except:
            return (None,None)

    def generate_files(self):
        for image_id in self.map:
            val=self.fetch_array(image_id)
            if val[0]:
                I=val[1]
                yield (image_id,I)
            else:
                tf.logging.info("Check image {}".format(image_id))
                continue

    def setup_data(self):
        self.ds=tf.data.Dataset.from_generator(self.generate_files, \
        output_types=(tf.string,tf.uint8), \
        output_shapes=(tf.TensorShape([]),tf.TensorShape([None,None,3])) )
        self.ds=self.ds.batch(self.batch_size)
        self.ds=self.ds.prefetch(20*self.batch_size)
        self.iter=self.ds.make_initializable_iterator()

    def build_inference_graph(self,image_tensor, graph_content):

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

        if self.batch_size==1:
            num_detections_tensor = tf.squeeze(
              g.get_tensor_by_name('num_detections:0'), 0)
        else:
            num_detections_tensor = g.get_tensor_by_name('num_detections:0')
            tf.logging.debug("NUM DETECTIONS TENSOR.{}".format(num_detections_tensor.shape))

        num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)

        if self.batch_size==1:
            detected_boxes_tensor = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), 0)
            detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]
        else:
            detected_boxes_tensor = g.get_tensor_by_name('detection_boxes:0')

        if self.batch_size==1:
            detected_scores_tensor = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), 0)
            detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]
        else:
            detected_scores_tensor = g.get_tensor_by_name('detection_scores:0')

        if self.batch_size==1:
            detected_labels_tensor = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), 0)
            detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)
            detected_labels_tensor = detected_labels_tensor[:num_detections_tensor]
        else:
            detected_labels_tensor = g.get_tensor_by_name('detection_classes:0')
            detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)

        return num_detections_tensor,detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor

    @staticmethod
    def check_flags(required_flags):
        for flag_name in required_flags:
          if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))


def main(_):
    start=time.time()

    tf.logging.set_verbosity(tf.logging.INFO)

    #Check FLAGS
    CellImagesRun.check_flags(required_flags = ['input_path','inference_graph'])

    # Intialise and load shard
    cell_run=CellImagesRun(FLAGS.input_path,FLAGS.batch_size)

    if FLAGS.test:
        cell_run.map=cell_run.map[:10]

    # Setup dataset object
    cell_run.setup_data()

    # Turn on log_device_placement for verbosity in ops
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
        sess.run(cell_run.iter.initializer)

        # Read and Fill Graph
        tf.logging.info('Reading graph and building model...')
        with tf.gfile.Open(FLAGS.inference_graph, 'rb') as graph_def_file:
            graph_content = graph_def_file.read()

        id,image_tensors=cell_run.iter.get_next()

        (num_detections_tensor,detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = cell_run.build_inference_graph(
             image_tensors, graph_content)

        try:
            for counter in itertools.count():
                tf.logging.info('Reading Image No; \t {} '.format(counter))

                tf.logging.info('Running inference')

                try:
                    (detected_boxes, detected_scores,detected_classes, num_detections,id_,image) = sess.run([detected_boxes_tensor,\
                     detected_scores_tensor,detected_labels_tensor,num_detections_tensor,id,image_tensors])
                except:
                    sess.run([id])
                    continue

                tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,counter)

                # Threshold SCORES
                ii=np.where(detected_scores>FLAGS.threshold)
                detected_scores=detected_scores[ii]
                detected_boxes=detected_boxes[ii]
                detected_classes=detected_classes[ii]

                tf.logging.debug("DETETED SCORES .{}".format((detected_scores)))
                tf.logging.debug("DETECTED BOXES {}".format(len(detected_boxes)))

                # Visulalise some high confidence images
                if FLAGS.vis_path and detected_scores.any():
                    if np.max(detected_scores)>FLAGS.vis_threshold:
                        image=np.squeeze(image)
                        # Draw boxes
                        id_=id_[0].decode("utf-8")
                        vis_utils.draw_bounding_boxes_on_image_array(image, detected_boxes)
                        vis_utils.save_image_array_as_png(image,os.path.join(FLAGS.vis_path,id_))

        except tf.errors.OutOfRangeError:
            # Catch exceptions
            tf.logging.info('Finished processing records')

        finally:
            end=time.time()
            tf.logging.info("Elapsed time {}".format(end-start))


if __name__ == '__main__':
    tf.app.run(main=main)
