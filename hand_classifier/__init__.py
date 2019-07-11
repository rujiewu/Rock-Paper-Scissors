import argparse
import datetime
import operator

import cv2
import numpy as np
from keras import backend as K
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model

from hand_classifier.utils import detector_utils as detector_utils
from hand_classifier import algorithm

K.set_image_dim_ordering('tf')

model = algorithm.HandRecognizer()


def detect(image):
    image = model.image_preprocess(image)
    gesture = model.guess_gesture(image)
    result = model.gesture_postprocess(gesture)
    return image, result
