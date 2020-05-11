"""
helper functions for predict.py
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import argparse

def process_image(image):
    """
    converts image (numpy array) to numpy array with shape (224,224,3)

    Parameters:
    image [np.array] - image

    Returns:
    image_processed [np.array] - reshaped image (224,224,3)
    """

    image_tensor = tf.convert_to_tensor(image)
    image_processed = tf.image.resize(image_tensor, size = (224,224))
    image_processed = image_processed / 255.
    image_processed = image_processed.numpy()

    return image_processed

def predict(image_path, model, top_k = 5):
    """
    Uses trained network for inference.

    Parameters:
    image_path [str]     - String
    model [Keras.model]  - Keras model
    top_k [int]          - number of n-highest values to return, default = 5

    Returns:
    classes [nd.array]   - top_k classes predicted, not zero-indexed!
    probs   [nd.array]   - top_k probabilities

    Requires:
    from PIL import Image
    """

    im = Image.open(image_path)
    image = np.asarray(im)

    image_processed = process_image(image)
    image_processed = np.expand_dims(image_processed, axis = 0)

    predictions = model.predict(image_processed)

    #get n-highest, negating values results in descending argsort
    #zero indexed classes!
    classes = np.argsort(-predictions[0])[:top_k]
    probs = np.asarray([predictions[0][classes]]).reshape(-1)
    #return non-zero indexed classes!
    classes = classes + 1

    return probs, classes
