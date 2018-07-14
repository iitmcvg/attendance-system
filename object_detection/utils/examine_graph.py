'''
Examine a given .pb file.

Syntax:

python object_detection/utils/examine_graph.py
--ckpt_path=/path/to/pb
'''

import tensorflow as tf

flags = tf.app.flags
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('ckpt_path', None,
                    'Path to a ckpt saved model.')

tf.app.flags.mark_flag_as_required('ckpt_path')

FLAGS = flags.FLAGS

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file(file_name=FLAGS.ckpt_path, tensor_name='', all_tensors=False, all_tensor_names=False)
