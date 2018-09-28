# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import numpy as np
from .tools import detect_face, get_model_filenames
from .Face_Alignment import  warp_im, coord5point,resizeimage


def detect(image_path):
        path1 = os.path.abspath('.')
        f_path = path1+'/MT/save_model/all_in_one'
        file_paths = get_model_filenames(f_path)

        imgresized = resizeimage(image_path)
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                config = tf.ConfigProto(allow_soft_placement=False)
                with tf.Session(config=config) as sess:
                        saver = tf.train.import_meta_graph(file_paths[0])
                        saver.restore(sess, file_paths[1])

                        def pnet_fun(imgresized): return sess.run(
                            ('softmax/Reshape_1:0',
                             'pnet/conv4-2/BiasAdd:0'),
                            feed_dict={
                                'Placeholder:0': imgresized})

                        def rnet_fun(imgresized): return sess.run(
                            ('softmax_1/softmax:0',
                             'rnet/conv5-2/rnet/conv5-2:0'),
                            feed_dict={
                                'Placeholder_1:0': imgresized})

                        def onet_fun(imgresized): return sess.run(
                            ('softmax_2/softmax:0',
                             'onet/conv6-2/onet/conv6-2:0',
                             'onet/conv6-3/onet/conv6-3:0'),
                            feed_dict={
                                'Placeholder_2:0': imgresized})

                        rectangles, points = detect_face(imgresized, 10,
                                                     pnet_fun, rnet_fun, onet_fun,
                                                     [0.8,0.8,0.8], 0.7)
                        points = np.transpose(points)

                        i = 0
                        L = []
                        for point in zip(rectangles,points):
                            i = i + 1
                            dst = warp_im(imgresized, np.array(point[1]).reshape(5, 2), coord5point)
                            crop_im = dst[0:112, 0:112]
                            L.append(crop_im)
                            # cv2.imwrite( "./"+ str(i) + ".jpeg", crop_im)

                        return L
