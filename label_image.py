# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import time
# from video import Video
# from fps import FPS
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_frame(frame,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  # input_name = "file_reader"
  # output_name = "normalized"
  # file_reader = tf.read_file(file_name, input_name)

  # if file_name.endswith(".png"):
  #   image_reader = tf.image.decode_png(
  #       file_reader, channels=3, name="png_reader")
  # elif file_name.endswith(".gif"):
  #   image_reader = tf.squeeze(
  #       tf.image.decode_gif(file_reader, name="gif_reader"))
  # elif file_name.endswith(".bmp"):
  #   image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  # else:
  #   image_reader = tf.image.decode_jpeg(
  #       file_reader, channels=3, name="jpeg_reader")

  float_caster = tf.convert_to_tensor(frame, dtype=tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def vidStream(url):
  print("[INFO] starting video stream...")
  vs = VideoStream(src=url).start()
  time.sleep(2.0)
  frame = vs.read() #frame is a numpy ndarray which is the return value of read_tensor_from_image_file
  return frame


if __name__ == "__main__":
  file_name = "/media/graymatics/backup/zachary/label/tensorflow/tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "/media/graymatics/backup/zachary/label/tensorflow/tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "/media/graymatics/backup/zachary/label/tensorflow/tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  # t = read_tensor_from_image_file(
  #     file_name,
  #     input_height=input_height,
  #     input_width=input_width,
  #     input_mean=input_mean,
  #     input_std=input_std)
  sess = tf.Session(graph=graph) # essentially sess = tf.Session(graph=graph) but session is closed after with blockprint("[INFO] starting video stream...")
  vs = VideoStream(src="http://192.168.1.152:8080").start()
  count = 0
  while(1):
    # start_time = time.time()
    # frame = vidStream("rtsp://admin:admin@192.168.1.137:30032") #slow af 3s
    # print("--- %s seconds ---" % (time.time() - start_time))
    frame = vs.read()
    # frame = frame[0:288, 0:299]
    # frame= cv2.resize(frame,(299,299),3)

    # t = np.expand_dims(frame, 0)
    # print(t.shape)
    # print(type(t))
    # t = tf.convert_to_tensor(frame, dtype=tf.float32)
    t= read_tensor_from_frame(frame)
    cv2.imshow("Frame", frame)
    # cv2.imwrite("./images/frame%d.jpg" % count, frame) 
    # count+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    # print(t.graph)
    # print(sess.grap)
    # t=t.eval(session=sess)
    # t =t.reshape(1, 299,299,3) # time = 0.004

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)


    print ("\nStarting object detection\n")
    
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
      print(labels[i], results[i])
    
  sess.close()



  	# for i in np.arange(0, detections.shape[2]):
		# # extract the confidence (i.e., probability) associated with
		# # the prediction
		# confidence = detections[0, 0, i, 2]
 
		# # filter out weak detections by ensuring the `confidence` is
		# # greater than the minimum confidence
		# if confidence > args["confidence"]:
		# 	# extract the index of the class label from the
		# 	# `detections`, then compute the (x, y)-coordinates of
		# 	# the bounding box for the object
		# 	idx = int(detections[0, 0, i, 1])
		# 	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 	(startX, startY, endX, endY) = box.astype("int")
 
		# 	# draw the prediction on the frame
		# 	label = "{}: {:.2f}%".format(CLASSES[idx],
		# 		confidence * 100)
		# 	cv2.rectangle(frame, (startX, startY), (endX, endY),
		# 		COLORS[idx], 2)
		# 	y = startY - 15 if startY - 15 > 15 else startY + 15
		# 	cv2.putText(frame, label, (startX, y),
		# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)