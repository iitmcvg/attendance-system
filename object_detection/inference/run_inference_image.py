"""Infers detections on an Image given an inference graph.

Example usage:
  ./run_inference_image.py \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb
    --image=/path/to/input_image.png
    --output_path=/path/to/output_image.png

The output is an image of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

All args:

./run_inference_image.py \
  --inference_graph=/path/to/frozen_weights_inference_graph.pb
  --image=/path/to/input_image.png
  --path_protofile=/path/to/protofile.pbtxt [for labels]
  --output_path=/path/to/output_image.png

"""

import itertools
import tensorflow as tf
import numpy as np

import PIL.Image as Image

from object_detection.inference import detection_inference
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import label_map_util

tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_string('image', None,
                       'Path to the input image.')
tf.flags.DEFINE_string('path_protofile', None,
                       'Path to the input image.')
tf.flags.DEFINE_string('output_path', None,
                       'Path to save output at. Extension must be png.')
tf.flags.DEFINE_float('confidence', 0.5,'Confidence to threshold with.')

FLAGS = tf.flags.FLAGS



def load_image(input_path):
    '''
    Load Image tensor

    Args:
    input_path: List of paths to the input image

    Returns:
    serialized_example_tensor: The next serialized example. String scalar Tensor
    image_tensor: The decoded image of the example. Uint8 tensor,
        shape=[1, None, None,3]

    '''
    with tf.gfile.GFile(input_path, 'rb') as fid:
        encoded_image = fid.read()

    image_tensor = tf.image.decode_image(encoded_image, channels=3)
    image_tensor.set_shape([None, None, 3])
    image_tensor = tf.expand_dims(image_tensor, 0)

    return image_tensor

def main(_):
    # Enable Verbose Logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # Check if all required flags are present
    required_flags = ['image', 'output_path',
                      'inference_graph']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))


    # Load category map
    '''
    A category index, which is a dictionary that maps integer ids to dicts
    containing categories, e.g.
    {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}
    '''

    category_index_from_labelmap=label_map_util.create_category_index_from_labelmap(FLAGS.path_protofile)

    with tf.Session() as sess:
        input_path=FLAGS.image
        tf.logging.info('Reading input from ', input_path)

        # Obtain image tensor
        image_tensor=load_image(input_path)

        # Run graph
        tf.logging.info('Reading graph and building model...')
        (detected_boxes_tensor, detected_scores_tensor,\
        detected_labels_tensor) = detection_inference.build_inference_graph(image_tensor, FLAGS.inference_graph)

        # Get detections
        (detected_boxes, detected_scores,\
        detected_labels)=sess.run([detected_boxes_tensor, detected_scores_tensor,\
        detected_labels_tensor])

        # Detected boxes of form: [ymins,xmins,ymax,xmax]

        input_image=sess.run(image_tensor)
        print(input_image)
        input_image=np.squeeze(input_image)

        # Draw bounding boxes
        print(detected_boxes,detected_scores)
        ii=np.where(detected_scores>FLAGS.confidence)
        for i in range(len(detected_scores[ii])):
            ymin=detected_boxes[i][0]
            xmin=detected_boxes[i][1]
            ymax=detected_boxes[i][2]
            xmax=detected_boxes[i][3]

            category=category_index_from_labelmap[detected_labels[i]]['name']

            vis_utils.draw_bounding_box_on_image_array(input_image,xmin=xmin,ymin=ymin,\
            xmax=xmax, ymax=ymax,display_str_list=(category) ,color='MediumPurple')

        vis_utils.save_image_array_as_png(input_image,FLAGS.output_path)


if __name__ == '__main__':
    tf.app.run()
