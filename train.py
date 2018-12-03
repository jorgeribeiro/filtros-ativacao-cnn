from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

from dataset_loader import DatasetLoader
from network_builder import NetworkBuilder
from constants import *

import plot_model

# This lines should be added on Windows (fuck Windows)
# No need to remove on Linux, works anyway
# The path to add is the graphviz's path on the machine
import os
os.environ['PATH'] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

# Load dataset
# Comment/uncomment to select dataset to use
dataset = DatasetLoader()

# JAFFE
# dataset.jaffe_load_from_save()
# num_classes = len(JAFFE_EMOTIONS)

# CK Extended
dataset.ck_extended_load_from_save()
num_classes = len(CK_EXTENDED_EMOTIONS)

# CK Extended no resize
# dataset.ck_extended_no_resize_load_from_save()
# num_classes = len(CK_EXTENDED_EMOTIONS)

# FERPlus (select one)
# dataset.fer_2013_small_load_from_save() # SMALL
# dataset.fer_2013_medium_load_from_save() # MEDIUM
# dataset.fer_2013_load_from_save() # FULL
# num_classes = len(FER_2013_EMOTIONS)

x_train, x_test = dataset.images, dataset.images_test
y_train, y_test = dataset.labels, dataset.labels_test

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# Build model
# Comment/uncomment to select model to use
network_builder = NetworkBuilder()
# model = VGG16(weights=None, input_shape=dataset.images.shape[1:], classes=num_classes)
model = InceptionV3(weights=None, input_shape=dataset.images.shape[1:], classes=num_classes)

# Compile
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # Compile for VGG-16 and InceptionV3

# Plot model for visualization
plot_model.plot(model, filename='regular_inception_v3')

# Train
DATA_AUG = False
epochs = 100
batch_size = 20
if not DATA_AUG:
	print ('[+] Training network without data augmentation')
	history = model.fit(
		x_train, y_train, 
		validation_data=(x_test, y_test),
		epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True).history
else:
	print ('[+] Training network with data augmentation')
	history = model.fit_generator(
		datagen.flow(x_train, y_train, batch_size=batch_size), 
		validation_data=(x_test, y_test),
		steps_per_epoch=len(x_train) // batch_size, epochs=epochs,
		verbose=1, shuffle=True).history
print ('[+] Training complete')

# Save model
model.save(MODELS_PATH + 'model_ck_extended_inception_v3_1.h5')
print ('[+] Model saved')