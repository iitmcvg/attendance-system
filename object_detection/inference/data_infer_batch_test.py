"""
Tests for data_infer_batch.py.

Usage:

python object_detection/inference/data_infer_batch_test.py \
--input_path="/media/ssd1/data/1040010018255600_jpg.tif_tiles/19/" \
--inference_graph="/media/ssd1/sat_data_models/faster_rcnn/large/export/frozen_inference_graph.pb" \
--output_path="a" \
--shard_path="" \
--batch_size=1 \
--test_size=10 \
--shard=0

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

tf.flags.DEFINE_boolean("test_images",True,"Whether to test IO")

FLAGS = tf.flags.FLAGS

import object_detection.inference.data_infer_batch as data_infer
import object_detection.utils.visualization_utils as vis_utils

# Check shards are vaild
assert(FLAGS.shard in range(4))

class SatellitleGeneratorTest(object):
    '''
    Test class for satellite generator.
    '''

    def __init__(self,path,batch_size=5,test_size=10):
        self.path=path
        self.batch_size=batch_size
        self.test_size=test_size
        self._init_x_y()

    def _init_x_y(self):

        # Find x coordinates
        xx =[f for f in glob.glob(os.path.join(self.path,'*')) if f[-3:]!='txt']
        xx=[x.split("/")[-1] for x in xx ]
        if not len(xx):
            raise ValueError('could not find any files in path: \t',self.path)

        # Find y coordinates
        yy = glob.glob(os.path.join(self.path,xx[0],'*.png'))
        if not len(yy):
            raise ValueError('could not find any files in path: . xx exists not yy \t',self.path)
        yy=[y.split("/")[-1][:-4] for y in yy ]

        tf.logging.debug("X{}".format(xx[:self.test_size]))
        tf.logging.debug("Y {}".format(yy[:self.test_size]))

        # Convert to integer
        xx=[int(x) for x in xx]
        yy=[int(y) for y in yy]

        # Split for testing
        size=self.test_size
        self.xx=xx[:size]
        self.yy=yy[:size]
        self.map=(self.xx,self.yy)

    def shard_x_y(self,shard_path):
        '''
        Shard the xx array into FLAGS.shard_size parts
        '''
        for i,j in zip(range(0,len(self.xx),int(len(self.xx)/FLAGS.shard_size)),range(FLAGS.shard_size)):
            path=os.path.join(shard_path,"shard"+str(j)+".npy")
            data=self.xx[i:i+int(len(self.xx)/FLAGS.shard_size)]
            tf.logging.debug("Size of shard {} is {}".format(j,len(data)))
            np.save(path, data)

    def check_for_shards(self,shard,shard_path):
        '''
        Check if all shards exist
        '''
        paths= [os.path.join(shard_path,"shard"+str(shard)+".npy") for shard in range(4)]
        exists= [os.path.exists(path) for path in paths]

        return all(exist is False for exist in exists )

    def load_shard(self,shard_path,shard=1):

        path=os.path.join(shard_path,"shard"+str(shard)+".npy")
        if not os.path.exists(path):
            tf.logging.debug("Shards not found, creating")
            self.shard_x_y(shard_path)
            self.xx=np.load(path)
        else:
            self.xx=np.load(path)
        self.map=(self.xx,self.yy)

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
                    I1=self.fetch_array(start1,path)
                else:
                    try:
                        I1=np.hstack((I1,self.fetch_array(start1,path)))
                    except Exception as e:
                        print(e)
                start1+=hor
            if i==0:
                I=I1
            else:
                I=np.vstack((I,I1))
            start=start+ver

        return I

    def read_patch_test(self,start,path):
        tf.logging.debug("PATCH {}".format(self._read_patch(start,path).shape))

    def generate_files(self):
        for x_i in range(0,len(self.map[0])-2,2):
            for y_i in range(0,len(self.map[1])-2,2):
                x=self.map[0][x_i]
                y=self.map[1][y_i]
                I=self._read_patch(np.array([x,y]),self.path)
                yield (x,y,I)

    def setup_data(self):
        self.ds=tf.data.Dataset.from_generator(self.generate_files, \
        output_types=(tf.int64,tf.int64,tf.uint8), \
        output_shapes=(tf.TensorShape(None),tf.TensorShape(None),tf.TensorShape([None,None,3])) )
        self.ds=self.ds.batch(self.batch_size)
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

    def write(self,x,y,size,detected_boxes,detected_scores,count=None):
        key=str(x.tolist()[0])+"/"+str(y.tolist()[0])
        sample={key:{'size':size,'detected_boxes':detected_boxes.tolist()\
        ,'detected_scores':detected_scores.tolist()}}

        if not os.path.exists(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json')) or  not count:
            with open(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json'),'w') as f:
                f.write(json.dumps(sample, indent=4))

        else:
            with open(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json'),'r+') as f:
                present=json.load(f)
                present.update(sample)
                f.seek(0, 0)
                f.write(json.dumps(present,indent=4))

def test(_):
    start=time.time()

    tf.logging.set_verbosity(tf.logging.DEBUG)

    #Check FLAGS
    SatellitleGeneratorTest.check_flags(required_flags = ['input_path','inference_graph'])

    # Intialise and load shard
    if FLAGS.test_size==-1:
        # Use default test size of 10
        sat_gen=SatellitleGeneratorTest(FLAGS.input_path,FLAGS.batch_size)
    else:
        sat_gen=SatellitleGeneratorTest(FLAGS.input_path,FLAGS.batch_size,FLAGS.test_size)
    sat_gen.load_shard(FLAGS.shard_path,FLAGS.shard)

    # Test a sample patch read
    coord=(sat_gen.map[0][0],sat_gen.map[1][0])
    sat_gen.read_patch_test(coord,FLAGS.input_path)

    # Setup dataset object
    sat_gen.setup_data()

    # Turn on log_device_placement for verbosity in ops
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
        sess.run(sat_gen.iter.initializer)

        if FLAGS.test_images:
            while True:
                x,y,image_tensors=sat_gen.iter.get_next()

            sys.exit(1)

        # Read and Fill Graph
        tf.logging.info('Reading graph and building model...')
        with tf.gfile.Open(FLAGS.inference_graph, 'rb') as graph_def_file:
            graph_content = graph_def_file.read()

        try:
            for counter in itertools.count():
                tf.logging.info('Reading Image No; \t {} SHARD{}'.format(counter,FLAGS.shard))
                x,y,image_tensors=sat_gen.iter.get_next()

                if not counter:
                    # Build Graph tensors only once
                    (num_detections_tensor,detected_boxes_tensor, detected_scores_tensor,
                     detected_labels_tensor) = sat_gen.build_inference_graph(
                         image_tensors, graph_content)


                tf.logging.info('Running inference')

                (detected_boxes, detected_scores,detected_classes, num_detections) = sess.run([detected_boxes_tensor,\
                 detected_scores_tensor,detected_labels_tensor,num_detections_tensor])

                tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,counter)

                # Threshold SCORES
                ii=np.where(detected_scores>FLAGS.threshold)
                detected_scores=detected_scores[ii]
                detected_boxes=detected_boxes[ii]
                detected_classes=detected_classes[ii]

                tf.logging.debug("DETETED SCORES .{}".format((detected_scores)))
                tf.logging.debug("DETECTED BOXES {}".format(len(detected_boxes)))

                x_,y_=sess.run([x,y])
                sat_gen.write(x_,y_,(1024,1024),detected_boxes,detected_scores,count=counter)

                # Visulalise some high confidence images
                if FLAGS.vis_path and detected_scores.any():
                    if np.max(detected_scores)>FLAGS.vis_threshold:
                        image=np.squeeze(sess.run(image_tensors))
                        # Draw boxes
                        vis_utils.draw_bounding_boxes_on_image_array(image, detected_boxes)
                        # Save
                        x_=x_[0]
                        y_=y_[0]
                        vis_utils.save_image_array_as_png(image,os.path.join(FLAGS.vis_path,str(x_)+"-"+str(y_)+".png"))

        except tf.errors.OutOfRangeError:
            # Catch exceptions
            tf.logging.info('Finished processing records')

        finally:
            end=time.time()
            tf.logging.info("Elapsed time {}".format(end-start))


if __name__=='__main__':
    tf.app.run(main=test)
