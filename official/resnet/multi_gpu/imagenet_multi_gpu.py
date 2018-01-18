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
"""Allows the Resnet-imagenet model to run multi-gpu.

TODO(karmel): When multi-GPU is out of contrib, use core instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import imagenet_main
import multi_gpu.resnet_multi_gpu as resnet_multi


def model_fn_with_optimizer_fn(features, labels, mode, params):
  """Wrapper for the model_fn that sets the optimizer function with the
  multi-GPU version..
  """
  return imagenet_main.resnet_model_fn(
    features, labels, mode, params, resnet_multi.get_optimizer)


def main(unused_argv):
  resnet_multi.main_with_model_fn(
      FLAGS, unused_argv, model_fn_with_optimizer_fn, imagenet_main.input_fn)


if __name__ == '__main__':
  parser = resnet_multi.ResnetMultiParser(
    resnet_size_choices=imagenet_main.VALID_SIZES)

  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
