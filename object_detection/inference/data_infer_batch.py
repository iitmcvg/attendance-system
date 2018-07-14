
r"""Infers detections on a Tile of satellite images given an inference graph.

Example usage:

 python object_detection/inference/data_infer_batch.py \
--input_path="/media/ssd1/data/1040010018255600_jpg.tif_tiles/19/" \
--inference_graph="/media/ssd1/sat_data_models/faster_rcnn/large/export/frozen_inference_graph.pb" \
--output_path="/media/ssd1/tile_output/" \
--shard_path="/media/ssd1/tile_output/" \
--batch_size=1 \
--shard=0 \
--vis_path="/media/ssd1/tile_output/"

"""

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
import functools

import json

import object_detection.utils.visualization_utils as vis_utils
from tensorflow.python.client import timeline

# String Flags
tf.flags.DEFINE_string('input_path', None,
                       'Tiled Satellite directory')
tf.flags.DEFINE_string('output_path', None,
                       'Path to the output json file.')
tf.flags.DEFINE_string('shard_path', '',
                       'Path to store sharded numpy arrays')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights and correct input batching.')
# Integer Flags
tf.flags.DEFINE_integer('batch_size', 1,
                       'Batch size for image runs per graph')
tf.flags.DEFINE_integer('test_size', -1,
                       'Test size')
tf.flags.DEFINE_integer('shard', 0,
                       'Shard to run')
tf.flags.DEFINE_integer('shard_size', 3,
                       'Shard size to use')

tf.flags.DEFINE_float('threshold', 0.1,
                       'Threshold to use for json')
tf.flags.DEFINE_float('vis_threshold', 0.9,
                       'Threshold to use for visualisation')

tf.flags.DEFINE_string('vis_path', '',
                       'Visualisation path with detections')

tf.flags.DEFINE_string('trace_path','trace','Path to store trace for profiling')

tf.flags.DEFINE_integer('write_every',200,'Pollling writting times')

tf.flags.DEFINE_boolean('restore_from_json',True,'Whether to restore from existent JSON files. ')

tf.flags.DEFINE_boolean('residue_run',True,'Whether to run a residual check. ')

tf.flags.DEFINE_string('total_covered','total_covered.npy','Numpy array for confirmed jpegs')

FLAGS = tf.flags.FLAGS

# Check shards are vaild
assert(FLAGS.shard in range(FLAGS.shard_size))

class SatellitleGenerator(object):
    '''
    Parse satellite tiles and run predictions.

    '''

    def __init__(self,path,batch_size=5,test_size=10):
        self.path=path
        self.batch_size=batch_size
        self.test_size=test_size
        self.residue_run=False
        self._init_x_y()

    def _init_x_y(self):

        # Find x coordinates
        xx =[f for f in glob.glob(os.path.join(self.path,'*')) if f[-3:] not in ['txt','kml']]
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

        self.xx=np.sort(self.xx)
        self.yy=np.sort(self.yy)

    @property
    def map(self):
        return (self.xx,self.yy)

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
            tf.logging.info("Shards not found, creating")
            self.shard_x_y(shard_path)
            self.xx=np.load(path)
        else:
            tf.logging.info("Shards found, loading from shard")
            self.xx=np.load(path)
        #self.map=(self.xx,self.yy)

    def load_residues(self,json_dir_path):
        '''
        Run on all residues present
        '''

        def _get_patches(path):
            start=path.split("/")
            start=[int(s) for s in start]
            start=np.array(start)
            hor=np.array([1,0])
            ver=np.array([0,1])
            paths=[]
            for i in range(4):
                for j in range(4):
                    paths.append(str(start[0])+"/"+str(start[1]))
                    start+=hor
                start=start+ver

            return paths    

        def _get_patches_ij(path,i,j):
            start=path.split("/")
            start=[int(s) for s in start]
            start=np.array(start)
            hor=np.array([1,0])
            ver=np.array([0,1])
            paths=[]
            for ii in range(i):
                for jj in range(j):
                    paths.append(str(start[0])+"/"+str(start[1]))
                    start+=hor
                start=start+ver

            return str(start[0])+"/"+str(start[1])       

        def _paths_uncovered(paths_covered,total_paths):
            tf.logging.info('paths covered so far {}'.format(len(paths_covered)))
            paths_covered=paths_covered[:,np.newaxis]
            for i in range(4): 
                for j in range(4):
                    if not i and not j:
                        paths_new=paths_covered
                    else:
                        functools.partial(_get_patches_ij,i=i,j=j)
                        vget_patches = np.vectorize(functools.partial(_get_patches_ij,i=i,j=j))
                        
                        paths_new=np.vstack((paths_new,vget_patches(paths_covered)))
                        tf.logging.info("mew trans {}".format(paths_new.shape))

            paths_covered=np.squeeze(paths_new)
            total_paths=np.array(total_paths)
            return total_paths[~np.isin(total_paths,paths_covered)]

        json_paths=[json_dir_path+"/result-"+str(i)+".json" for i in range(2)]

        total_paths=glob.glob(self.path+"/*/*.png")
        total_paths=["/".join(i.split("/")[-2:])[:-4] for i in total_paths]

        data={}
        for path in json_paths:
            with open(path,'r') as f:
                data.update(json.load(f))

        paths_covered=np.array(list(data.keys()))

        total_paths=_paths_uncovered(paths_covered,total_paths)

        tf.logging.info("Total paths left {}".format(total_paths))
        tf.logging.info("No of total paths left {}".format(len(total_paths)))

        self.residue_run=True
        self.total_paths=total_paths

    
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
        covered=[]
        for i in range(4):
            start1=start.copy()
            for j in range(4):
                if j==0:
                    # Intialise I1
                    try:
                        I1=self.fetch_array(start1,path)
                        covered.append(str(start[0])+"/"+str(start[1]))
                    except:
                        I1=np.zeros((256,256,3))
                        covered.append(str(start[0])+"/"+str(start[1])+"-Missed")
                else:
                    try:
                        I1=np.hstack((I1,self.fetch_array(start1,path)))
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

    def read_patch_test(self,start,path):
        tf.logging.debug("PATCH {}".format(self._read_patch(start,path)[0].shape))

    def generate_files(self):
        if self.residue_run:
            for path in self.total_paths:
                path=path.split("/")
                x=int(path[0])
                y=int(path[1])
                I,covered=self._read_patch(np.array([x,y]),self.path)
                if np.any(I):
                    yield (x,y,I,covered)
                else:
                    continue
        
        else:
            for x_i in range(0,len(self.map[0])-3,2):
                for y_i in range(0,len(self.map[1])-3,2):
                    x=self.map[0][x_i]
                    y=self.map[1][y_i]
                    I,covered=self._read_patch(np.array([x,y]),self.path)
                    if np.any(I):
                        yield (x,y,I,covered)
                    else:
                        continue

    def setup_data(self):
        self.ds=tf.data.Dataset.from_generator(self.generate_files, \
        output_types=(tf.int64,tf.int64,tf.uint8,tf.string), \
        output_shapes=(tf.TensorShape(None),tf.TensorShape(None),tf.TensorShape([None,None,3]),tf.TensorShape([None])) )
        #self.ds=self.ds.batch(self.batch_size)
        self.ds=self.ds.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
        self.ds=self.ds.prefetch(10*self.batch_size)
        self.iter=self.ds.make_initializable_iterator()

    def restore_from_json(self,json_path):
        '''
        Restores from going over list of operated images.
        '''
        with open(json_path,'r') as f:
            data=json.load(f)

        try:
            xx_neglect=np.array([int(s.split("/")[0]) for s in data.keys()])
            tf.logging.info("Shape {}".format(xx_neglect.shape))
            yy_neglect=np.array([int(s.split("/")[1]) for s in data.keys()])

        except Exception as e:
            print(e)
            s_valid=[]
            for s in data.keys():
                try:
                    int(s.split("/")[0])
                    s_valid.append(s)
                except:
                    continue
            data={s:data[s] for s in s_valid}
            with open(json_path,'w') as f:
                json.dump(data,f)
            xx_neglect=np.array([int(s.split("/")[0]) for s in data.keys()])
            yy_neglect=np.array([int(s.split("/")[0]) for s in data.keys()])

        # Adding other skipped images
        xx_neglect=np.unique(xx_neglect)
        yy_neglect=np.unique(yy_neglect)
        tf.logging.info("{} XX matches {}".format(FLAGS.shard,len(xx_neglect)))
        tf.logging.info("{} YY matches {}".format(FLAGS.shard,len(yy_neglect)))
        xx_neglect=np.expand_dims(xx_neglect,axis=1)
        yy_neglect=np.expand_dims(yy_neglect,axis=1)
        xx_neglect=np.vstack((xx_neglect,xx_neglect+1))
        yy_neglect=np.vstack((yy_neglect,yy_neglect+1))

        # Create masks
        self.xx=np.array(self.xx)
        mask1=np.isin(self.xx,xx_neglect,invert=True)

        self.yy=np.array(self.yy)
        mask2=np.isin(self.yy,yy_neglect,invert=True)

        if not np.any(mask1) or not np.any(mask2):
            tf.logging.info("{} Nothing found in JSON.".format(FLAGS.shard))
        else:
            images_done=(len(xx_neglect)-3)*(len(yy_neglect)-3)
            images_total=(len(self.xx)-3)*(len(self.yy) -3)/4
            tf.logging.info("{} Finished xx's {} , yy's {} , images {}".format(FLAGS.shard,len(xx_neglect),len(yy_neglect),images_done))
            tf.logging.info("{} Of total xx's {} , yy's {} , images {}".format(FLAGS.shard,len(self.xx),len(self.yy),images_total))

        self.yy=self.yy[mask2]
        self.xx=self.xx[mask1]

        tf.logging.info("{} Reduced sizes XX's {} YY's {}".format(FLAGS.shard,len(self.xx),len(self.yy)))

        if not(np.any(self.xx)) and not(np.any(self.yy)):
            return "Finished"

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

        if (not os.path.exists(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json')) or  not count) and not FLAGS.restore_from_json and not FLAGS.residue_run:
            with open(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json'),'w') as f:
                f.write(json.dumps(sample, indent=4))

        else:
            with open(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json'),'r+') as f:
                present=json.load(f)
                present.update(sample)
                f.seek(0, 0)
                f.write(json.dumps(present,indent=4))

    @staticmethod
    def check_flags(required_flags):
        for flag_name in required_flags:
          if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))


def main(_):
    start=time.time()

    tf.logging.set_verbosity(tf.logging.INFO)

    #Check FLAGS
    SatellitleGenerator.check_flags(required_flags = ['input_path','inference_graph'])

    # Intialise and load shard
    sat_gen=SatellitleGenerator(FLAGS.input_path,FLAGS.batch_size,FLAGS.test_size)

    if FLAGS.residue_run:
        print("residue run")
        sat_gen.load_residues(FLAGS.output_path)

    sat_gen.load_shard(FLAGS.shard_path,FLAGS.shard)

    if FLAGS.restore_from_json:
        message=sat_gen.restore_from_json(os.path.join(FLAGS.output_path,'result-'+str(FLAGS.shard)+'.json'))

        if message:
            tf.logging.info("Finished all evaluations here. Exitting")
            didnotrun=True
            sys.exit(0)
        else:
            didnotrun=False
    else:
        didnotrun=False
    # Test a sample patch read
    '''
    coord=(sat_gen.map[0][0],sat_gen.map[1][0])
    sat_gen.read_patch_test(coord,FLAGS.input_path)
    '''
    # Setup dataset object
    sat_gen.setup_data()

    # Session configs
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    # Turn on log_device_placement for verbosity in ops

    # Covered png's
    total_covered=[]

    with tf.Session(config=config) as sess:
        sess.run(sat_gen.iter.initializer)

        # Read and Fill Graph
        tf.logging.info('Reading graph and building model...')
        with tf.gfile.Open(FLAGS.inference_graph, 'rb') as graph_def_file:
            graph_content = graph_def_file.read()

        x,y,image_tensors,covered=sat_gen.iter.get_next()

        (num_detections_tensor,detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = sat_gen.build_inference_graph(
             image_tensors, graph_content)

        # Chrome tracing
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        try:
            for counter in itertools.count():
                tf.logging.info('Reading Image No; \t {} SHARD{}'.format(counter,FLAGS.shard))

                tf.logging.info('Running inference')

                if counter%10==0:
                    (detected_boxes, detected_scores, num_detections,x_,y_,image,covered_) = sess.run([detected_boxes_tensor,\
                     detected_scores_tensor,num_detections_tensor,x,y,image_tensors,covered],
                     options=options, run_metadata=run_metadata)

                else:
                    (detected_boxes, detected_scores, num_detections,x_,y_,image,covered_) = sess.run([detected_boxes_tensor,\
                     detected_scores_tensor,num_detections_tensor,x,y,image_tensors,covered])

                tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,counter)

                total_covered.append(covered_)

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
                    tf.logging.info("Writting to JSON SHARD {}".format(FLAGS.shard) )
                    sat_gen.write(x_stack,y_stack,(1024,1024),detected_boxes_stack,detected_scores_stack,count=counter,batch_size=FLAGS.batch_size,stack_size=FLAGS.write_every)
                    x_stack=x_
                    y_stack=y_
                    detected_boxes_stack=detected_boxes
                    detected_scores_stack=detected_scores
                    np.save(FLAGS.total_covered,np.array(total_covered).flatten().astype(str))

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
                '''
                # Chrome Profiling trace
                if counter%100==0:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(os.path.join(FLAGS.trace_path,'timeline_'+str(counter)+'.json'), 'w') as f:
                        f.write(chrome_trace)
                '''

        except tf.errors.OutOfRangeError:
            # Catch exceptions
            tf.logging.info('Finished processing records')

        finally:
            if didnotrun:
                pass
            else:
                # print(x_stack.shape)
                sat_gen.write(x_stack,y_stack,(1024,1024),detected_boxes_stack,detected_scores_stack,batch_size=FLAGS.batch_size,stack_size=int(len(x_stack)/FLAGS.batch_size),count=True)
                tf.logging.info('Finished writting residual stacks of size {}'.format(len(x_stack)))
            end=time.time()
            tf.logging.info("Elapsed time {}".format(end-start))


if __name__ == '__main__':
    tf.app.run(main=main)
