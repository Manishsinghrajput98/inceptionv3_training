import argparse
from flask import Flask, request, redirect, jsonify
from pykakasi import kakasi
import skimage
import tensorflow as tf
import mrcnn.model as modellib
import numpy as np
import os
import uuid
import cv2

class Manishxyz123(Flask):
    def __init__(self, host, name,classification_label,sess,softmax_tensor):
        super(Manishxyz123, self).__init__(name,static_url_path='')
        self.host = host
        self.define_uri()
        self.requests = {}
        self.class_name = classification_label
        self.sess = sess
        self.softmax_tensor = softmax_tensor
        self.result = None

    def define_uri(self):
        self.provide_automatic_option = False
        self.add_url_rule('/start', None, self.start,
                          methods=['POST'])

    def setup_converter(self):
        mykakasi = kakasi()
        mykakasi.setMode('H', 'a')
        mykakasi.setMode('K', 'a')
        mykakasi.setMode('J', 'a')
        self.converter = mykakasi.getConverter()

    def display_instances(self,image_path):
        frame = skimage.io.imread(image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        n_id = top_k[0]
        labell = self.class_name[n_id]
        for node_id in top_k:
            label = self.class_name[node_id]
            score = predictions[0][node_id]
            if score*100 >= 70:
                text = "Activity: {} ".format(label)
                self.result = label
                cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 3)
                random_name = str(uuid.uuid4())
                skimage.io.imsave('output/' + random_name + ".jpg" ,frame)
        return self.result

    def start(self):
        print(("json :",request.get_json()))
        if request.method == 'POST':
            body = request.get_json()
            print(("body :: {}".format(body)))
            image_path = body['image_path']
            result = self.display_instances(image_path)        
            res = dict()
            res['status'] = '200'
            res['result'] = result
            print('====iinceptionv3model====',res)
            return jsonify(res)
            	
def importargs():
    parser = argparse.ArgumentParser('This is a server of inceptionv3')
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=8080)
    args = parser.parse_args()
    return args.host, args.port

def main():
    host, port  = importargs()
    classification_label = [line.rstrip() for line in tf.gfile.GFile('logs/train.txt')]
    with tf.gfile.FastGFile('logs/train.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        graph = tf.get_default_graph()
        sess = tf.Session()
        softmax_tensor = graph.get_tensor_by_name('final_result:0')
    server = Manishxyz123(host, 'inceptionv3_model',classification_label,sess,softmax_tensor,)
    server.run(host=host, port=port)
if __name__ == "__main__":
    main()


