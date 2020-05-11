"""
used as command line application according:
The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

Usage:
$ python predict.py /path/to/image saved_model --category_names map.json

Arguments:
/path/to/image [str] - path to image file
saved_model [str] - path to h5 model file

Options:
--top_k [int] - number of top classes shown
--category_names [JSON] - mapping if classes<->labels
"""

# Make all necessary imports.
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import os
import argparse
import helpers as h

#add arguments
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to the image file")
parser.add_argument("saved_model", help="Keras model file")
parser.add_argument("--top_k", type = int, help="Top_k number of top classes shown")
parser.add_argument("--category_names", help="JSON file with class<->label mapping")
args = parser.parse_args()

#extract arguments
image_path = args.image_path
model_file = args.saved_model

#handle optional arguments
if args.top_k:
    top_k = args.top_k
else:
    top_k = 5

class_names = {}
if args.category_names:
    category_names = args.category_names
    with open(category_names, 'r') as f:
        class_names = json.load(f)  #dict


#load model
model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})

#predict
probs, classes = h.predict(image_path, model, top_k)

#print out results
print(f"probabilities: {probs}")
if class_names:
    classes = [class_names[str(n)] for n in classes]
print(f"classes: {classes}")


"""
for testing and debugging
python predict.py ./test_images/wild_pansy.jpg test_model.h5
python predict.py ./test_images/wild_pansy.jpg test_model.h5 --top_k 9 --category_names label_map.json
python predict.py ./test_images/wild_pansy.jpg test_model.h5 --top_k 2
python predict.py ./test_images/wild_pansy.jpg test_model.h5 --category_names label_map.json

python predict.py ./test_images/hard-leaved_pocket_orchid.jpg image_net_oxford_flowers_20200511_174947_0.81.h5
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg image_net_oxford_flowers_20200511_174947_0.81.h5 --top_k 10
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg image_net_oxford_flowers_20200511_174947_0.81.h5 --category_names label_map.json
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg image_net_oxford_flowers_20200511_174947_0.81.h5 --category_names label_map.json --top_k 10
"""
