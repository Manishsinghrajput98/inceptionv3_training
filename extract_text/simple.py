from PIL import Image
import pytesseract
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True)

args = vars(ap.parse_args())

input_images = args["input"]

Text = pytesseract.image_to_string(Image.open(input_images))

print ("output of the images:-",Text)

