from PIL import Image
import pytesseract
import argparse
import cv2 
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True)

args = vars(ap.parse_args())

input_images = args["input"]

img = cv2.imread(input_images)

kernel = np.ones((2, 2), np.uint8)/5

img = cv2.dilate(img, kernel, iterations=1)

img = cv2.erode(img, kernel, iterations=0)

img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

Text = pytesseract.image_to_string(img)

print ("output of the images:-",Text)