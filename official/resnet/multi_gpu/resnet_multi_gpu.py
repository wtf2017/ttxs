#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Shared functionality for running resnet models on different
data sets, multi-GPU.

TODO(karmel): When multi-GPU is out of contrib, use core instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import resnet_main
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_optimizer(learning_rate, momentum):
  """Wrapper for getting the optimizer for multi-gpu.
  """
  optimizer = resnet_main.get_optimizer(learning_rate, momentum)
  optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
  return optimizer


def main_with_model_fn(flags, unused_argv, model_fn, input_fn):
  replicated_fn = tf.contrib.estimator.replicate_model_fn(
      model_fn, loss_reduction=tf.losses.Reduction.MEAN)
  resnet_main.main_with_model_fn(flags, unused_argv, replicated_fn, input_fn)


# TODO(karmel): Working in parallel with MNIST right now,
# but should be shared code with that.
def get_reasonable_batch_size(current_size):
  """For multi-gpu, batch-size must be a multiple of the number of
  available GPUs.

  TODO(karmel): This should eventually be handled by replicate_model_fn
  directly. For now, doing the work here.
  """
  devices = _get_local_devices('GPU') or _get_local_devices('CPU')
  num_devices = len(devices)
  remainder = current_size % num_devices
  return current_size - remainder


# TODO(karmel): Replicated from
# tf.contrib.estimator.python.estimator.replicate_model_fn . Should not
# be a copy, but to avoid import problems until this is done by
# replicate_model_fn itself, including here.
def _get_local_devices(device_type):
  local_device_protos = device_lib.list_local_devices()
  return [
      device.name
      for device in local_device_protos
      if device.device_type == device_type
  ]


class ResnetMultiParser(resnet_main.ResnetArgParser):
  def __init__(self, resnet_size_choices=None):
    super(ResnetMultiParser, self).__init__(resnet_size_choices)
    # Set default batch size
    batch_size = get_reasonable_batch_size(self.get_default('batch_size'))
    self.set_defaults(batch_size=batch_size)
