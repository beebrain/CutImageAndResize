from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from sklearn.metrics import classification_report,confusion_matrix

from keras import backend as K
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import Finger
import PrintPredicTable

nb_classes = 25

# load json and create model
json_file = open('CNN_model_MLP.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("CNN_model_MLP.h5")
print("Loaded model from disk")

# HOG
# trainHog = np.load("TrainHog.npy")
# testHog = np.load("TestHog.npy")
# Y_train = np.load("Y_train.npy")
# Y_test = np.load("Y_test.npy")


#CNN
input_X_train = np.load("CNNTrain.npy")
input_X_test=  np.load("CNNTest.npy")
Y_train = np.load("CNN_Ytrain.npy")
Y_test = np.load("CNN_Ytest.npy")
print "Loaded.."

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
e = loaded_model.predict_classes(input_X_test)
PrintPredicTable.printTable(np.where(Y_test)[1],e,nb_classes)
