import argparse
from flask import Flask, request, redirect, jsonify
from pykakasi import kakasi
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import skimage
import colorsys
import tensorflow as tf
import mrcnn.model as modellib
import numpy as np
import os 
import shutil
import random
import uuid

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class TestConfig(Config):
    NAME = "mask_rcnn"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
config = TestConfig()

class Manishxyz123(Flask):
    def __init__(self, host, name,graph,mask_model):
        super(Manishxyz123, self).__init__(name,static_url_path='')
        self.host = host
        self.define_uri()
        self.requests = {}
        self.mask_model = mask_model
        self.graph = graph

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

    def random_colors(self,N, bright=True):
        brightness = 0.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(self,image, mask, color,bounding_box,j):
        alpha=0.0
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,image[:, :, c] *(1 - alpha) + alpha * color[c] * 255,image[:, :, c])
        image = image.copy()
        image[mask == 0] = (255, 255, 255)
        random_name = str(uuid.uuid4())
        skimage.io.imsave('output/' + random_name + ".jpg" ,image)

    def display_instances(self,image_path):
        final = []
        img = skimage.io.imread(image_path)
        with self.graph.as_default():
            r = self.mask_model.detect([img], verbose=0)[0]
        image = img
        boxes = r['rois']
        masks = r['masks']
        ids =  r['class_ids']
        scores = r['scores']
        names = class_names
        n_instances = boxes.shape[0]
        if not n_instances:
            print('----Not_Detected----')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        for j,i in enumerate(range(n_instances)):
              if not np.any(boxes[i]):
                  continue
              y1, x1, y2, x2 = boxes[i]
              label = names[ids[i]]
              colors = self.random_colors(len(class_names))
              class_dict = {
                 name: color for name, color in zip(class_names, colors)
              }
              color = class_dict[label]
              score = scores[i] if scores is not None else None
              caption = '{} {:.2f}'.format(label, score) if score else label
              mask = masks[:, :, i]
              bounding_box = (x1, y1, x2, y2)
              self.apply_mask(image, mask, color, bounding_box,j)
              final.append(str(label))
        return final

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
            print('====Mask_Rcnn_Detection_Result====',res)
            return jsonify(res)
            	
def importargs():
    parser = argparse.ArgumentParser('This is a server of mask rcnn')
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=8080)
    args = parser.parse_args()
    return args.host, args.port

def main():
    host, port  = importargs()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./tree')
    model.load_weights('mask_rcnn_coco.h5', by_name=True)
    mask_model = model
    graph = tf.get_default_graph()
    server = Manishxyz123(host, 'Mask_Rcnn_Detection_Model',graph,mask_model)
    server.run(host=host, port=port)
if __name__ == "__main__":
    main()




