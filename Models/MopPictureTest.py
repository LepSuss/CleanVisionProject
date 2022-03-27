## This can be used after you have used the Moptest.py to train your model
## This loads the saved model weights to this identical model and quickly evaluates
## through all of the Final_Mop_test pictures (the original pictures provided from the original meeting in November)

## Useful for actually seeing how your model handles pictures it's never seen before

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D

import pathlib

batch_size = 32
img_height = 256
img_width = 256

num_classes = 5

path = "C:\\Users\\leevi\\Jupyter\\Python Codes\\Mops"

data_dir = pathlib.Path(path)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  seed=112,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

chck_filepath = '/Users/leevi/Jupyter/tmp/checkpoint'

model.load_weights(chck_filepath)

# remember to change right directory for where your final test pictures are
directory = 'C:/Users/leevi/Downloads/Final_Mop_test'

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        mop_path = os.path.join(directory, filename)
        im = PIL.Image.open(mop_path)
        im.show()

        img = keras.preprocessing.image.load_img(
            mop_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This " + filename + " most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )