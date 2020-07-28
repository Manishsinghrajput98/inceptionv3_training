from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("-video", "--input", required=True,
	help="path to our input video")
ap.add_argument("-out", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

classification_label = [line.rstrip() for line in tf.gfile.GFile('logs/trained_labels.txt')]

with tf.gfile.FastGFile('logs/trained_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    graph = tf.get_default_graph()
    sess = tf.Session()
    softmax_tensor = graph.get_tensor_by_name('final_result:0')
    print('classificaiton graph load')

Q = deque(maxlen=args["size"])
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	cv2.imwrite("temp.jpg",frame)
	framee = "temp.jpg"
	image_data = tf.gfile.FastGFile(framee, 'rb').read()
	print('===============classify_image start==================')
	result = []
	predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	n_id = top_k[0]
	labell = classification_label[n_id]
	for node_id in top_k:
	    label = classification_label[node_id]
	    score = predictions[0][node_id]
	    result.append({'label': label, 'score': str(score)})
	    print(('%s (score = %.5f)' % (label, score*100)))
	    result.append([score*100,label])
	print(result)
	Q.append(result)
	img = cv2.imread("temp.jpg")
	text = "activity: {}".format(labell)
	cv2.putText(img, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
	writer.write(img)
writer.release()
vs.release()
