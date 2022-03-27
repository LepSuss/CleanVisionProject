# Just used for creating augmented pictues to databank

import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = "C:\\Users\\leevi\\tmp\\Mops3\\UnUsable"

data_dir = pathlib.Path(path)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

input("Press Enter to continue...")

batch_size = 8
img_width = 512
img_height = 512

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        brightness_range=(0.2,0.8),
        channel_shift_range=80,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[0.5,1.0],
        validation_split=0.2)
train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(512, 512),
        batch_size=16,
        seed=55,
        class_mode='binary',
        save_to_dir="C:\\Users\\leevi\\tmp\\augmented\\train",
        save_format="jpeg",
        subset="training"
        )
validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(512, 512),
        batch_size=16,
        seed=55,
        class_mode='binary',
        save_to_dir="C:\\Users\\leevi\\tmp\\augmented\\validation",
        save_format="jpeg",
        subset="validation")

num_classes = 2

model = Sequential([
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

epochs=20
history = model.fit(
        train_generator,
        steps_per_epoch=1,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()