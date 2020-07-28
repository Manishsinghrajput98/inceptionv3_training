from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import tensorflow as tf
import os
import argparse
import uuid


ap = argparse.ArgumentParser()
ap.add_argument("-image", "--input", required=True)
args = vars(ap.parse_args())

image = args["input"]

classification_label = [line.rstrip() for line in tf.gfile.GFile('logs/trained_labels.txt')]
with tf.gfile.FastGFile('logs/trained_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    graph = tf.get_default_graph()
    sess = tf.Session()
    softmax_tensor = graph.get_tensor_by_name('final_result:0')
    print('classificaiton graph load')

frame = cv2.imread(image)
image_data = tf.gfile.FastGFile(image, 'rb').read()
predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
n_id = top_k[0]
labell = classification_label[n_id]
for node_id in top_k:
	label = classification_label[node_id]
	score = predictions[0][node_id]
	print(('%s (score = %.5f)' % (label, score*100)))
	text = "Activity: {} ".format(labell)
	cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 3)
	random_name = str(uuid.uuid4())
cv2.imwrite(random_name + ".jpg", frame) 
