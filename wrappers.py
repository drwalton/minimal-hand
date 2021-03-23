from collections import namedtuple
from typing import Tuple, Any

import numpy as np
import tensorflow as tf

from config import HAND_MESH_MODEL_PATH, IK_UNIT_LENGTH, DETECTION_MODEL_PATH, IK_MODEL_PATH
from kinematics import mano_to_mpii, xyz_to_delta, MPIIHandJoints
from network import detnet, iknet
from utils import load_pkl

FRACTION = 0.95 / 4
view = ['xyz', 'uv', 'hmap']  # + ['feed_img', 'dmap', 'lmap', 'hmap_flat', 'argmax', 'argmax_x', 'argmax_y']
IntermVals = namedtuple('IntermVals', view)


class ModelDet:
  """
  DetNet: estimating 3D keypoint positions from input color image.
  """
  def __init__(self, model_path):
    """
    Parameters
    ----------
    model_path : str
      Path to the trained model.
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.compat.v1.variable_scope('prior_based_hand'):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FRACTION
        self.sess = tf.compat.v1.Session(config=config)
        self.input_ph = tf.compat.v1.placeholder(tf.uint8, [128, 128, 3])
        self.feed_img = tf.cast(tf.expand_dims(self.input_ph, 0), tf.float32) / 255
        self.hmaps, self.dmaps, self.lmaps = detnet(self.feed_img, 1, False)

        self.hmap = self.hmaps[-1]
        self.dmap = self.dmaps[-1]
        self.lmap = self.lmaps[-1]

        self.hmap_flat = tf.reshape(self.hmap, (tf.shape(self.hmap)[0], -1, tf.shape(self.hmap)[3]))
        self.argmax = tf.argmax(self.hmap_flat, axis=1, output_type=tf.int32)
        self.argmax_x = self.argmax // tf.shape(self.hmap)[2]
        self.argmax_y = self.argmax % tf.shape(self.hmap)[2]
        self.uv = tf.stack((self.argmax_x, self.argmax_y), axis=1)
        self.uv = tf.transpose(self.uv, [0, 2, 1])

        self.delta = tf.gather_nd(
          tf.transpose(self.dmap, [0, 3, 1, 2, 4]), self.uv, batch_dims=2
        )[0]
        self.xyz = tf.gather_nd(
          tf.transpose(self.lmap, [0, 3, 1, 2, 4]), self.uv, batch_dims=2
        )[0]

        self.uv = self.uv[0]
      tf.compat.v1.train.Saver().restore(self.sess, model_path)

  def process(self, img):
    """
    Process a color image.

    Parameters
    ----------
    img : np.ndarray
      A 128x128 RGB image of **left hand** with dtype uint8.

    Returns
    -------
    np.ndarray, shape [21, 3]
      Normalized keypoint locations. The coordinates are relative to the M0
      joint and normalized by the length of the bone from wrist to M0. The
      order of keypoints is as `kinematics.MPIIHandJoints`.
    np.ndarray, shape [21, 2]
      The uv coordinates of the keypoints on the heat map, whose resolution is
      32x32.
    """
    results = self.sess.run([getattr(self, name) for name in view], {self.input_ph: img})
    return IntermVals(*results)


class ModelIK:
  """
  IKnet: estimating joint rotations from locations.
  """
  def __init__(self, input_size, network_fn, model_path, net_depth, net_width):
    """
    Parameters
    ----------
    input_size : int
      Number of joints to be used, e.g. 21, 42.
    network_fn : function
      Network function from `network.py`.
    model_path : str
      Path to the trained model.
    net_depth : int
      Number of layers.
    net_width : int
      Number of neurons in each layer.
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.input_ph = tf.compat.v1.placeholder(tf.float32, [1, input_size, 3])
      with tf.name_scope('network'):
        self.theta = \
          network_fn(self.input_ph, net_depth, net_width, training=False)[0]
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FRACTION
        self.sess = tf.compat.v1.Session(config=config)
      tf.compat.v1.train.Saver().restore(self.sess, model_path)

  def process(self, joints):
    """
    Estimate joint rotations from locations.

    Parameters
    ----------
    joints : np.ndarray, shape [N, 3]
      Input joint locations (and other information e.g. bone orientation).

    Returns
    -------
    np.ndarray, shape [21, 4]
      Estimated global joint rotations in quaternions.
    """
    theta = self.sess.run(self.theta, {self.input_ph: np.expand_dims(joints, 0)})
    if len(theta.shape) == 3:
      theta = theta[0]
    return theta


class ModelPipeline:
  """
  A wrapper that puts DetNet and IKNet together.
  """
  def __init__(self):
    # load reference MANO hand pose
    mano_ref_xyz = load_pkl(HAND_MESH_MODEL_PATH)['joints']
    # convert the kinematic definition to MPII style, and normalize it
    mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
    mpii_ref_xyz -= mpii_ref_xyz[9:10]
    # get bone orientations in the reference pose
    mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
    mpii_ref_delta = mpii_ref_delta * mpii_ref_length

    self.mpii_ref_xyz = mpii_ref_xyz
    self.mpii_ref_delta = mpii_ref_delta

    self.det_model = ModelDet(DETECTION_MODEL_PATH)
    # 84 = 21 joint coordinates
    #    + 21 bone orientations
    #    + 21 joint coordinates in reference pose
    #    + 21 bone orientations in reference pose
    self.ik_model = ModelIK(84, iknet, IK_MODEL_PATH, net_depth=6, net_width=1024)

  def process(self, frame) -> Tuple[IntermVals, Any]:
    """
    Process a single frame.

    Parameters
    ----------
    frame : np.ndarray, shape [128, 128, 3], dtype np.uint8.
      Frame to be processed.

    Returns
    -------
    np.ndarray, shape [21, 3]
      Joint locations.
    np.ndarray, shape [21, 4]
      Joint rotations.
    """
    interm_vals = self.det_model.process(frame)
    delta, length = xyz_to_delta(interm_vals.xyz, MPIIHandJoints)
    delta *= length
    pack = np.concatenate(
      [interm_vals.xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta], axis=0
    )
    theta = self.ik_model.process(pack)

    return interm_vals, theta
