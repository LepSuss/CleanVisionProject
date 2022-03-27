# THis model Overfits way too fast

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
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import preprocess_input

#------------------ Fetch dataset from disk and create train + val datasets -------------------------------------------------

path = "C:\\Users\\leevi\\Jupyter\\Python Codes\\Mops2" 

data_dir = pathlib.Path(path)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Tried to get all the parameters here :)
batch_size = 32
IMG_SIZE = (224, 224)
num_classes = 2
IMG_SHAPE = IMG_SIZE + (3,)

base_learning_rate = 0.0001
initial_epochs = 4
fine_tune_epochs = 3
fine_tune_at = 80

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=27,
  image_size= IMG_SIZE,
  batch_size= batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  shuffle=True,
  seed=27,
  image_size= IMG_SIZE,
  batch_size= batch_size)

class_names = train_dataset.class_names
print(class_names)

#-------------------- Adding prefetch for faster performance --------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#--------------------- Data augmentation for picture variety ------------------------------

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

#------------------ Setting the MobileNetV2 as the base model --------------------------

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#--------------------- Naming few layers and creating our model ------------------------------
# So basically we are creating a small model that is going to be on top of the MobileNet model

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2, activation='softmax')                                              

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x= layers.Dense(512,activation='relu')(x) 
x= layers.Dense(512,activation='relu')(x) 
x= layers.Dense(256,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# We are only training our own layers and leaving the MobileNet alone (for now)
for layer in base_model.layers:
    layer.trainable=False

#--------------------------- Compiling the model --------------------------------------

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
plt.ylim([min(plt.ylim()),1])
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

base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

#------------------------ Compiling the model again so the changes take effect --------------------------------------
# Notice the change in learning rate!

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate/50),
              metrics=['accuracy'])

#----------------------------- Adding checkpoint path and configuring callbacks --------------------------------------------

chck_filepath = '/Users/leevi/Jupyter/tmp/Mops3Checkpoint/Model/checkpoint'

my_callbacks = [
    #EarlyStopping(monitor="val_accuracy",patience=5, mode='max', restore_best_weights=True),
    ModelCheckpoint(filepath=chck_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True),
]

model.summary()

#------------------------------ Training the model again / Fine-tuning -----------------------------------------------------
# So now we are training some of the MobileNets' layers along with our own layers to get a better fit for our data

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
plt.ylim([0.8, 1])
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

directory = 'C:/Users/leevi/Downloads/Final_Mop_test/Grey'

test_path = "C:/Users/leevi/Downloads/Final_Mop_test/test_mops"

data_dir2 = pathlib.Path(test_path)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir2,
  image_size= IMG_SIZE,
  batch_size= 6)

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

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
        score = tf.nn.softmax(predictions[0])

        print(
            "This " + filename + " most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

save_folder = '/Users/leevi/Jupyter/tmp/Mops3Checkpoint/Model/Full'

model.save(save_folder)