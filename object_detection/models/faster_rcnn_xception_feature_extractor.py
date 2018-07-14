
# Custom definition

"""Xception model.

"Xception: Deep Learning with Depthwise Separable Convolutions"
Fran{\c{c}}ois Chollet
https://arxiv.org/abs/1610.02357

We implement the modified version by Jifeng Dai et al. for their COCO 2017
detection challenge submission, where the model is made deeper and has aligned
features for dense prediction tasks. See their slides for details:

"Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge
2017 Entry"
Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei and Jifeng Dai
ICCV 2017 COCO Challenge workshop
http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

We made a few more changes on top of MSRA's modifications:
1. Fully convolutional: All the max-pooling layers are replaced with separable
  conv2d with stride = 2. This allows us to use atrous convolution to extract
  feature maps at any resolution.

2. We support adding ReLU and BatchNorm after depthwise convolution, motivated
  by the design of MobileNetv1.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications"
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, Hartwig Adam
https://arxiv.org/abs/1704.04861
"""

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import xception

slim = tf.contrib.slim

class FasterRCNNXceptionFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Xception feature extractor implementation."""

  def __init__(self,
               architecture,
               xception_model,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      architecture: Architecture name of the Xception model.
      xception_model: Definition of the Xception model.
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._architecture = architecture
    self._xception_model = xception_model
    super(FasterRCNNXceptionFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Xception preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    channel_means = [123.68, 116.779, 103.939]
    return resized_inputs - [[channel_means]]

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the

       size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      # Disables batchnorm for fine-tuning with smaller batch sizes.
      # TODO(chensun): Figure out if it is needed when image
      # batch size is bigger.
      with slim.arg_scope(
          xception.xception_arg_scope(
              batch_norm_epsilon=1e-3,
              batch_norm_scale=True,
              weight_decay=self._weight_decay)):
        with tf.variable_scope(
            self._architecture, reuse=self._reuse_weights) as var_scope:
          _, activations = self._xception_model(
              preprocessed_inputs,
              num_classes=None,
              is_training=self._train_batch_norm,
              global_pool=False,
              output_stride=self._first_stage_features_stride,
              scope=var_scope)

    handle = scope + '/%s/exit_flow/block1' % self._architecture
    return activations[handle], activations

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
      with slim.arg_scope(
          xception.xception_arg_scope(
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              weight_decay=self._weight_decay)):
        with slim.arg_scope([slim.batch_norm],
                            is_training=self._train_batch_norm):
          blocks = [
              xception.Block('exit_flow/block2',xception.xception_module,[{
                  'depth_list':[1536, 1536, 2048],
                  'skip_connection_type':'none',
                  'activation_fn_in_separable_conv':True,
                  'regularize_depthwise':False,
                  'stride':1,
                  'unit_rate_list':[1, 1, 1]
              }] * 1)
          ]
          proposal_classifier_features = xception.stack_blocks_dense(
              proposal_feature_maps, blocks)
    return proposal_classifier_features


class FasterRCNNXception41FeatureExtractor(FasterRCNNXceptionFeatureExtractor):
  """Faster R-CNN Xception 41 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNXception41FeatureExtractor, self).__init__(
        'xception_41', xception.xception_41, is_training,
        first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)


class FasterRCNNXception65FeatureExtractor(FasterRCNNXceptionFeatureExtractor):
  """Faster R-CNN Xception 65 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNXception65FeatureExtractor, self).__init__(
        'xception_65', xception.xception_65, is_training,
        first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)


class FasterRCNNXception71FeatureExtractor(FasterRCNNXceptionFeatureExtractor):
  """Faster R-CNN Xception 71 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNXception71FeatureExtractor, self).__init__(
        'xception_71', xception.xception_71, is_training,
        first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)
