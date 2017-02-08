'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
#
from keras import backend as K
import os
import cv2
import matplotlib.pyplot as plt
import Finger
import VisualizeNN as VSN
import time
print(os.path.expanduser('~'))
batch_size = 100
nb_classes = 25
nb_epoch = 400
data_augmentation = True

# input image dimensions
img_rows, img_cols = 64, 64
# the CIFAR10 images are RGB
img_channels = 1

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = Finger.load_data(False,personalselected=5)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


print (K.image_dim_ordering())
if K.image_dim_ordering() == 'th':
    input_X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    input_X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
    print("theno")
else:
    input_X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
    input_X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,img_channels)
    input_shape = (img_rows, img_cols, img_channels)
    print("tersorflow")

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

if os.path.isfile("CNN_model_MLP.json") :
    # load json and create model
    json_file = open('CNN_model_MLP.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("CNN_model_MLP.h5")
    print("Loaded model from disk")

    acc = np.load("history.npy")
    acc = acc.tolist()
    print("load history")

else :
    print("create model")
    #create model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #history value
    acc = []

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

input_X_train = input_X_train.astype('float32')
input_X_test = input_X_test.astype('float32')
input_X_train /= 255
input_X_test /= 255

np.save("CNNTrain.npy",input_X_train)
np.save("CNNTest.npy",input_X_test)
np.save("CNN_Ytrain.npy",Y_train)
np.save("CNN_Ytest.npy",Y_test)

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(input_X_train)

vsn = VSN.Visualplot()

# fit the model on the batches generated by datagen.flow()

for epochin in xrange(nb_epoch):
    history = model.fit_generator(datagen.flow(input_X_train, Y_train,
                    batch_size=batch_size),
                    samples_per_epoch=input_X_train.shape[0],
                    nb_epoch=1,
                    validation_data=(input_X_test, Y_test))

    if acc == []:
        acc = history.history
        print ("new acc")
    else:
        acc['acc'].extend(history.history['acc'])
        acc['val_acc'].extend( history.history['val_acc'])
        acc['loss'].extend(history.history['loss'])
        acc['val_loss'].extend(history.history['val_loss'])
        print("append acc")

    if epochin%3 == 0:
        # serialize model to JSON
        model_json = model.to_json()
        with open("CNN_model_MLP.json", "w") as json_file:  json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("CNN_model_MLP.h5")
        print("Saved model to disk")

        #save history
        np.save("history.npy",acc)
        vsn.plot(acc)


# serialize model to JSON
model_json = model.to_json()
with open("CNN_model_MLP.json", "w") as json_file: json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNN_model_MLP.h5")
print("Saved model to disk")

# load json and create model
json_file = open('CNN_model_MLP.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("CNN_model_MLP.h5")
print("Loaded model from disk")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()