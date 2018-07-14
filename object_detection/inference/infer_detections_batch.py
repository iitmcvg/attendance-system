
r"""Infers detections on a Tile of satellite images given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import itertools
import tensorflow as tf

from object_detection.inference import detection_inference

tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

FLAGS = tf.flags.FLAGS

class Satellite(object):
    '''
    Read, pre-process and run on satellite images

    '''


    def fetch_array(coordinate,image_path):
        raise NotImplementedError

    def read_patch(path,start):
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
        I[256:512,:256]=image(start+(1,0))
        I[256:512,:256]=image(start+(1,0))

        '''

        hor=np.array([1,0])
        ver=np.array([0,1])

        for i in range(4):
            start1=start

            for

            I1=np.hstack(fetch_array(start,path))

            I=np.vstack((I,I1))
        pass

    def build_inference_graph(image_tensor, graph_content):

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

        num_detections_tensor = tf.squeeze(
          g.get_tensor_by_name('num_detections:0'), 0)
        num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)

        detected_boxes_tensor = tf.squeeze(
          g.get_tensor_by_name('detection_boxes:0'), 0)
        detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]

        detected_scores_tensor = tf.squeeze(
          g.get_tensor_by_name('detection_scores:0'), 0)
        detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]

        detected_labels_tensor = tf.squeeze(
          g.get_tensor_by_name('detection_classes:0'), 0)
        detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)
        detected_labels_tensor = detected_labels_tensor[:num_detections_tensor]

        return detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor





def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  sat=Satellite()

  sat.check_flags(required_flags = ['input_tfrecord_paths', 'output_tfrecord_path','inference_graph'])



  with tf.Session() as sess:
    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]

    number,image_tensor = detection_inference.build_input(
        input_tfrecord_paths)

    tf.logging.info('Reading Queue \t',number)

    tf.logging.info('Reading graph and building model...')

    with tf.gfile.Open(FLAGS.inference_graph, 'rb') as graph_def_file:
        graph_content = graph_def_file.read()

    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = build_inference_graph(
         image_tensor, graph_content)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()

    (serialized_example, detected_boxes, detected_scores,
     detected_classes) = sess.run([
         serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor
     ])
    detected_boxes = detected_boxes.T

    with tf.python_io.TFRecordWriter(
        FLAGS.output_tfrecord_path) as tf_record_writer:
      try:
        for counter in itertools.count():
          tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                 counter)
          tf_example = detection_inference.infer_detections_and_add_to_example(
              serialized_example_tensor, detected_boxes_tensor,
              detected_scores_tensor, detected_labels_tensor,
              FLAGS.discard_image_pixels)
          tf_record_writer.write(tf_example.SerializeToString())
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished processing records')


if __name__ == '__main__':
  tf.app.run()
