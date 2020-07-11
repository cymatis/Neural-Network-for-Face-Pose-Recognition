# incompatable with tf < 2.0.0b or use tf-nightly (2.2.0 causes None shape error)
import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import datetime

####################################################

# path
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'Face dataset_png/train')
validation_dir = os.path.join(base_dir, 'Face dataset_png/val')

# constants
batch_size = 32
epochs = 100
IMG_HEIGHT = 30
IMG_WIDTH = 32
input_shape = (32,30)

# pre-declare where the image tuples will be contained
train_image_generator = ImageDataGenerator(rescale=1./255) # rescale image byte, from 0-255 to 0-1
validation_image_generator = ImageDataGenerator(rescale=1./255) # rescale image byte, from 0-255 to 0-1

# make tuple of images and labels for train, (batch_size, image_info, label)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, # directory of train images
                                                           shuffle=True, # shuffle when starts new epoch
                                                           batch_size = batch_size,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), # expected image size from directory
                                                           color_mode = 'grayscale', # declare that 8bit png grayscale image will be used as input
                                                           class_mode='categorical') # this output(tuple) generate multiple labels

# make tuple of images and labels for validation, (batch_size, image_info, label)
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir, # directory of validation images
                                                              shuffle=True, # shuffle when starts new epoch
                                                              batch_size = batch_size,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH), # expected image size from directory
                                                              color_mode = 'grayscale', # declare that 8bit png grayscale image will be used as input
                                                              class_mode='categorical') # this output(tuple) generate multiple labels

####################################################

# Keras model by using tensorflow API

model = Sequential([
    Flatten(input_shape=(IMG_HEIGHT,IMG_WIDTH,1)),# Flatten layer will get images as input and vectorize them, The Glorot normal initializer, also called Xavier normal initializer.
    Dense(3, activation='sigmoid', kernel_initializer='glorot_normal'), # use relu for activation func.
    Dense(4, activation='sigmoid', kernel_initializer='glorot_normal') # output needs 4 types of face direction
])

# using SGD for training optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.3,
                              momentum=0.3)

# compile Keras model defined above
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# show compiled model structure and trainable values
model.summary() 

####################################################

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1,
                                                      write_images=True)

# tensorboard --logdir logs/fit << run this command on seperate terminal

####################################################

# start training and save training information to histroy
history = model.fit(train_data_gen, # train dataset
                    epochs=epochs, # epochs
                    validation_data=val_data_gen, # validation dataset
                    callbacks=[tensorboard_callback]
)

####################################################

if not(os.path.isdir(os.path.join(base_dir, 'models'))):
    os.makedirs(os.path.join(os.path.join(base_dir, 'models')))

model_dir = os.path.join(base_dir, 'models')
model.save(model_dir)

####################################################

weight_file_1 = os.path.join(base_dir, 'weight_file_01.txt')
weight_file_2 = os.path.join(base_dir, 'weight_file_02.txt')

f = open(weight_file_1, 'wt')
f2 = open(weight_file_2, 'wt')

weight_raw = model.layers[1].get_weights()[0]  #the first layer
weight_raw2 = model.layers[2].get_weights()[0]  #the second layer

f.write(np.array_str(weight_raw))
f2.write(np.array_str(weight_raw2))

f.close()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

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

####################################################

print(sample_weight)