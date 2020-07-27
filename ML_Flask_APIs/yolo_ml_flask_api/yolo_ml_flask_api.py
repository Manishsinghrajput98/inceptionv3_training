import argparse
from flask import Flask, request, redirect, jsonify
from pykakasi import kakasi
import numpy as np
import os
import cv2
import time

class ManishXYZ123(Flask):
    def __init__(self, host, name,model,classes):
        super(ManishXYZ123, self).__init__(name,static_url_path='')
        self.host = host
        self.define_uri()
        self.requests = {}
        self.model = model
        self.classes = classes
        self.conf_threshold = 0.5
        self.nms_threshold = 0.3
        self.scale = 0.00392

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

    def image_detection(self,image_path):
        image = cv2.imread(image_path)
        Width = image.shape[1]
        Height = image.shape[0]
        COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        ln = self.model.getLayerNames()
        output_layers = [ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
                    for i in indices:
                        i = i[0]
                        box = boxes[i]
                        x = box[0]
                        y = box[1]
                        w = box[2]
                        h = box[3]
                        x1=int(round(x))
                        y1=int(round(y))
                        x_plus_w=int(round(x+w))
                        y_plus_h=int(round(y+h))
                        label = str(self.classes[class_id])
                        color = COLORS[class_id]
                        cv2.rectangle(image,(x1,y1),(x_plus_w,y_plus_h), color, 3)
                        cv2.putText(image, label, (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.imwrite("result.jpg" , image)
        return label
        
    def start(self):
        print(("json :",request.get_json()))
        if request.method == 'POST':
            body = request.get_json()
            print(("body :: {}".format(body)))
            image_path = body['image_path']
            result = self.image_detection(image_path)        
            res = dict()
            res['status'] = '200'
            res['result'] = result
            print('====YOLO_Detection_Result====',res)
            return jsonify(res)
            	
def importargs():
    parser = argparse.ArgumentParser('This is a server of mask rcnn')
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=8080)
    args = parser.parse_args()
    return args.host, args.port

def main():
    host, port  = importargs()
    model = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    server = ManishXYZ123(host, 'YOLO_Detection_Model',model,classes)
    server.setup_converter()
    server.run(host=host, port=port)

if __name__ == "__main__":
    main()




