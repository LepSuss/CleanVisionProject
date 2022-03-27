# Final model probably works on pure luck as it gives very differing results on different runs.
# Validation accuracy stagnates around 60 %, but on the best run the difference between new and 900 is around 0.19 out of 0 to 1, which is the best we have
# Trained model is usually heavy towards UnUsable, but the predictions can be fixed and use by changing the classification threshold

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D

#------------------ Fetch dataset from disk and create train + val datasets -------------------------------------------------

path = "C:\\Users\\leevi\\tmp\\Mops3Orig" 

data_dir = pathlib.Path(path)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Tried to get all the parameters here :)
batch_size = 8
IMG_SIZE = (224, 224)
num_classes = 2
IMG_SHAPE = IMG_SIZE + (3,)

base_learning_rate = 0.01
initial_epochs = 50
fine_tune_epochs = 50
fine_tune_at = 30

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
train_dataset = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        seed=62,
        class_mode='binary',
        subset="training"
        )
validation_dataset = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        seed=62,
        class_mode='binary',
        subset="validation")

class_names = ["UnUsable", "Usable"]

#------------------ Setting the EfficientNet as the base model --------------------------

base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#--------------------- Naming few layers and creating our model ------------------------------
# So basically we are creating a small model that is going to be on top of the EfficientNet model

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

preprocess_input = preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#feature_batch_average = global_average_layer(feature_batch)
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
#prediction_batch = prediction_layer(feature_batch_average)                                             

inputs = tf.keras.Input(shape=(224, 224, 3))
x = inputs
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# We are only training our own layers and leaving the EfficientNet alone (for now)
for layer in base_model.layers:
    layer.trainable=False

#--------------------------- Compiling the model --------------------------------------

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              metrics=['accuracy'])

model.summary()

#------------------ Training the model and plotting the results ------------------------

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0.3,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#----------- Changing the base model to trainable and setting the amount of bottom layers to be left alone -------------

#base_model.trainable = True

for layer in base_model.layers[-fine_tune_at:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable =  True

#------------------------ Compiling the model again so the changes take effect --------------------------------------
# Notice the change in learning rate!

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate/100),
              metrics=['accuracy'])

#----------------------------- Adding checkpoint path and configuring callbacks --------------------------------------------

chck_filepath = '/Users/leevi/tmp/Mops3Checkpoint/checkpoint'

my_callbacks = [
    #EarlyStopping(monitor="val_accuracy",patience=5, mode='max', restore_best_weights=True),
    ModelCheckpoint(filepath=chck_filepath, save_weights_only=True, monitor='accuracy', mode='max', save_best_only=True),
]

model.summary()

#------------------------------ Training the model again / Fine-tuning -----------------------------------------------------
# So now we are training some of the EfficientNet' layers along with our own layers to get a better fit for our data

total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         callbacks=my_callbacks)

#------------------------- Some plotting --------------------------------------------

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.3, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#loading the best weights back from the saved callback checkpoint, before final testing.
model.load_weights(chck_filepath)


#---------------------- Below is some testing stuff -----------------------------------------


scores = model.evaluate(validation_dataset, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

save_folder = '/Users/leevi/tmp/Mops3Checkpoint/Modeltest'

model.save(save_folder)

directory = 'C:/Users/leevi/tmp/test_directory'

test_path = "C:/Users/leevi/tmp/test_mops"

data_dir2 = pathlib.Path(test_path)

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        mop_path = os.path.join(directory, filename)
        im = PIL.Image.open(mop_path)
        im.show()

        img = keras.preprocessing.image.load_img(
            mop_path, target_size=(IMG_SIZE)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        print(predictions)
        predictions = predictions[0]
        #predictions = predictions.numpy()
        print(predictions)

        score = np.max(predictions)
        print(score)

        result = tf.where(predictions < 0.5, 0, 1)
        result = np.max(result.numpy())

        if result == 0:
          score = 1-np.max(predictions)

        print(
            "This " + filename + " most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[result], score*100)
        )