#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--use-vg3k',
        dest='use_vg3k',
        help='use Visual Genome 3k classes (instead of COCO 80 classes)',
        action='store_true'
    )
    parser.add_argument(
        '--thresh',
        default=0.7,
        type=float,
        help='score threshold for predictions',
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


class kinect_segmenter:
    def callback(self, data):
        self.logger.info('Processing {} -> {}')
        #        print("converting ros img to cv2 and passing it to maskrcnn")
        self.bridge = CvBridge()
        im = self.bridge.imgmsg_to_cv2(data, "bgr8")
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, im, None, timers=timers
            )
        self.logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            self.logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        result = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            "my_test",
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=self.dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=2
        )
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(result[0], "bgr8"))
            self.pub_bbox.publish(result[1].astype(np.float32))
        except CvBridgeError as e:
            print(e)

    def __init__(self, args):
        self.image_pub = rospy.Publisher("/segmented_mug_rgb", Image, queue_size=10)
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.callback)
        self.pub_bbox = rospy.Publisher('bbox', numpy_msg(Floats), queue_size=10)
        self.logger = logging.getLogger(__name__)
        merge_cfg_from_file(args.cfg)
        cfg.NUM_GPUS = 1
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)
        self.model = infer_engine.initialize_model_from_cfg(args.weights)
        self.dummy_coco_dataset = (
            dummy_datasets.get_vg3k_dataset()
            if args.use_vg3k else dummy_datasets.get_coco_dataset())


def main(args):
    kc = kinect_segmenter(args)
    rospy.init_node('segment_kinect', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
