import numpy as np
import cv2

from dataset_loader import DatasetLoader

from constants import *

from keras.models import load_model
from quiver_engine.server import launch

# Load dataset
# Comment/uncomment to select dataset to use
dataset = DatasetLoader()

# CK Extended
dataset.ck_extended_load_from_save()
classes = CK_EXTENDED_EMOTIONS
num_classes = len(classes)

x_train, x_test = dataset.images, dataset.images_test
y_train, y_test = dataset.labels, dataset.labels_test

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Load model
model = load_model(MODELS_PATH + 'model_ck_extended_vgg_16_2.h5')
print ('[+] Model loaded')

count = 0
result = model.predict(x_test)
for i in range(len(result)):	
	if (classes[np.argmax(result[i])] == classes[np.argmax(y_test[i])]):
		count += 1

print ('[+] Validation accuracy: ', count / len(result))
