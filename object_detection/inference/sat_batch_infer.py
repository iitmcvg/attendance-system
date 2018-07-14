
r"""Infers detections on a Tile of satellite images given an inference graph.

Example usage:

 python object_detection/inference/sat_batch_infer.py \
--input_path="/media/ssd1/data/1040010018255600_jpg.tif_tiles/19/" \
--inference_graph="/media/ssd1/sat_data_models/faster_rcnn/large/export/frozen_inference_graph.pb" \
--output_path="/media/ssd1/tile_output/" \
--batch_size=16 \
--vis_path="/media/ssd1/tile_output/"

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
import sys
import json

# Module imports 
import object_detection.utils.visualization_utils as vis_utils
from tensorflow.python.client import timeline

# String Flags
tf.flags.DEFINE_string('input_path', None,
                       'Tiled Satellite directory')
tf.flags.DEFINE_string('output_path', None,
                       'Path to the output json file.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights and correct input batching.')
tf.flags.DEFINE_string('vis_path', '',
                       'Visualisation path with detections')

# Integer Flags
tf.flags.DEFINE_integer('batch_size', 1,
                       'Batch size for image runs per graph')
tf.flags.DEFINE_integer('test_size', -1,
                       'Test size')
tf.flags.DEFINE_integer('write_every',200,'Pollling writting times')


# Float Flags
tf.flags.DEFINE_float('threshold', 0.1,
                       'Threshold to use for json')
tf.flags.DEFINE_float('vis_threshold', 0.9,
                       'Threshold to use for visualisation')

# Boolean Flags
tf.flags.DEFINE_boolean('restore_from_json',True,'Whether to restore from existent JSON files. ')

FLAGS = tf.flags.FLAGS

class SatellitleGenerator(object):
    '''
    Parse satellite tiles and run predictions.
    '''

    def __init__(self,path,json_path,batch_size=5,test_size=10):
        self.path=path
        self.json_path=json_path
        self.batch_size=batch_size
        self.test_size=test_size
        self._init_paths()

    def _init_paths(self):
        '''
        initialise paths
        '''
         def _check(coord):
            '''
            coord of form x/y
            '''
            x=int(coord.split("/")[0])
            y=int(coord.split("/")[1])

            # Check if both are even
            return (x%2==0 and y%2==0)

        # Find x coordinates
        map=glob.glob(os.path.join(self.path,'*','*.png')))
        map=["/".join(s.split("/")[-2:])[:-4] for s in map]

        # Filter only even coordinates
        map=[mapcoord for mapcoord in map if _check(mapcoord)]
        map=np.array(map)
        map=map.sort()

        tf.logging.info("Found map size of {}".format(len(map)))

        self.map=map
        self.map=self.map[:self.test_size]

    def load_progress(self):
        '''
        Skip present detections.
        '''
        with open(self.json_path,'r') as f:
            data=json.load(f)
        
        paths_done=np.array(list(data.keys()))
        self.map=self.map[~np.isin(self.map,paths_done)]

    def fetch_array(self,coordinate,image_path):
        '''
        Read image array from path
        '''
        path=os.path.join(image_path,str(coordinate[0]),str(coordinate[1])+".png")
        return scipy.ndimage.imread(path,mode="RGB").astype(np.uint8)

    def _read_patch(self,start,path):
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

        for i in range(4):
            start1=start.copy()
            for j in range(4):
                if j==0:
                    # Intialise I1
                    try:
                        I1=self.fetch_array(start1,path)
                    except:
                        I1=np.zeros((256,256,3))
                else:
                    try:
                        I1=np.hstack((I1,self.fetch_array(start1,path)))
                    except Exception as e:
                        I1=np.hstack((I1,np.zeros((256,256,3))))
                start1+=hor
            if i==0:
                I=I1
            else:
                I=np.vstack((I,I1))
            start=start+ver

        return I

    def generate_files(self):
        '''
        Generator for dataset iterator
        '''
        for x_i in range(0,len(self.map[0])-3,2):
            for y_i in range(0,len(self.map[1])-3,2):
                x=self.map[0][x_i]
                y=self.map[1][y_i]
                I=self._read_patch(np.array([x,y]),self.path)
                if np.any(I):
                    yield (x,y,I)
                else:
                    continue

    def setup_data(self):
        '''
        Makes dataset iterator
        '''
        self.ds=tf.data.Dataset.from_generator(self.generate_files, \
        output_types=(tf.int64,tf.int64,tf.uint8), \
        output_shapes=(tf.TensorShape(None),tf.TensorShape(None),tf.TensorShape([None,None,3])) )
        self.ds=self.ds.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
        self.ds=self.ds.prefetch(10*self.batch_size)
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
        tf.import_graph_def(graph_def, name='', input_map={'image_tensor': image_tensor})
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

    def write(self,x,y,size,detected_boxes,detected_scores,count=None,batch_size=1,stack_size=1):
        sample={}
        '''
        detected_boxes shape: (batch_size*stack_size,100,4)
        detected_scores shape: (batch_size*stack_size,100)
        '''
        for i in range(batch_size*stack_size):
            try:
                key=str(x[i])+"/"+str(y[i])

                # Threshold SCORES
                ii=np.where(detected_scores[i]>FLAGS.threshold)
                d_s=detected_scores[i][ii]
                d_b=detected_boxes[i][ii]

                tf.logging.debug("DETETED SCORES .{}".format((d_s)))
                tf.logging.debug("DETECTED BOXES {}".format(len(d_b)))

                sample[key]={'size':size,'detected_boxes':d_b.tolist()\
                ,'detected_scores':d_s.tolist()}
            except Exception as e:
                tf.logging.info("Error e (could be due to residual writting)".format(e))
                continue

        if (not os.path.exists(self.json_path) or  not count):
            with open(self.json_path,'w') as f:
                f.write(json.dumps(sample, indent=4))

        else:
            with open(self.json_path,'r+') as f:
                present=json.load(f)
                present.update(sample)
                f.seek(0, 0)
                f.write(json.dumps(present,indent=4))

    @staticmethod
    def check_flags(required_flags):
        '''
        Throws error when flag is absent.
        '''
        for flag_name in required_flags:
          if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

def main(_):
    start=time.time()
    tf.logging.set_verbosity(tf.logging.INFO)

    #Check FLAGS
    SatellitleGenerator.check_flags(required_flags = ['input_path','inference_graph'])
    # Intialise instance
    sat_gen=SatellitleGenerator(FLAGS.input_path,os.path.join(FLAGS.output_path,"result.json")\
    ,FLAGS.batch_size,FLAGS.test_size)

    if os.path.exists(sat_gen.json_path) and FLAGS.restore_from_json:
        tf.logging.info("Loading from JSON path at {}".format(sat_gen.json_path))
        sat_gen.load_progress()

    # Setup dataset object
    sat_gen.setup_data()

    # Session configs
    # Turn on log_device_placement for verbosity in ops
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(sat_gen.iter.initializer)

        # Read and Fill Graph
        tf.logging.info('Reading graph and building model...')
        with tf.gfile.Open(FLAGS.inference_graph, 'rb') as graph_def_file:
            graph_content = graph_def_file.read()

        x,y,image_tensors=sat_gen.iter.get_next()

        (num_detections_tensor,detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = sat_gen.build_inference_graph(
             image_tensors, graph_content)

        try:
            for counter in itertools.count():
                tf.logging.info('Reading Image No; \t {} from {}'.format(counter,FLAGS.input_path))
                tf.logging.info('Running inference')
                
                (detected_boxes, detected_scores, num_detections,x_,y_,image) = sess.run([detected_boxes_tensor,\
                    detected_scores_tensor,num_detections_tensor,x,y,image_tensors])

                tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,counter)

                if counter==0:
                    # Initialise stacks
                    tf.logging.info("Initialsied stacks")
                    x_stack=x_
                    y_stack=y_
                    detected_boxes_stack=detected_boxes
                    detected_scores_stack=detected_scores

                # Calibrate to write at counter = FLAGS.write_every-1, 2*FLAGS.write_every-1, ...
                if (counter)%FLAGS.write_every==FLAGS.write_every-1 :
                    # Write to JSON
                    tf.logging.info("Writting to JSON FILE at {}".format(sat_gen.json_path) )
                    sat_gen.write(x_stack,y_stack,(1024,1024),detected_boxes_stack,detected_scores_stack,count=counter,batch_size=FLAGS.batch_size,stack_size=FLAGS.write_every)
                    x_stack=x_
                    y_stack=y_
                    detected_boxes_stack=detected_boxes
                    detected_scores_stack=detected_scores
                else:
                    x_stack=np.hstack((x_stack,x_))
                    y_stack=np.hstack((y_stack,y_))
                    detected_boxes_stack=np.vstack((detected_boxes_stack,detected_boxes))
                    detected_scores_stack=np.vstack((detected_scores_stack,detected_scores))

                # Visulalise some high confidence images
                if FLAGS.vis_path and detected_scores.any():
                    if np.max(detected_scores)>FLAGS.vis_threshold:
                        for j in range(FLAGS.batch_size):
                            if np.max(detected_scores[j])>FLAGS.vis_threshold:
                                # Draw boxes
                                boxes_ii=np.where(detected_scores[j]>FLAGS.vis_threshold)
                                boxes=detected_boxes[j][boxes_ii]
                                vis_utils.draw_bounding_boxes_on_image_array(image[j],boxes)
                                # Save
                                vis_utils.save_image_array_as_png(image[j],os.path.join(FLAGS.vis_path,str(x_[j])+"-"+str(y_[j])+".png"))

        except tf.errors.OutOfRangeError:
            # Catch exceptions
            tf.logging.info('Finished processing records')

        finally:
            if len(sat_gen.map):
                sat_gen.write(x_stack,y_stack,(1024,1024),detected_boxes_stack,detected_scores_stack,batch_size=FLAGS.batch_size,stack_size=int(len(x_stack)/FLAGS.batch_size),count=True)
                tf.logging.info('Finished writting residual stacks')
            end=time.time()
            tf.logging.info("Elapsed time {}".format(end-start))

if __name__ == '__main__':
    tf.app.run(main=main)
