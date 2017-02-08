from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras import backend as K
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import Finger

import ImageAssert as IMA

#  prepair Data
batch_size = 1000
nb_classes = 25
nb_epoch = 400
data_augmentation = True

# # the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = Finger.load_data(False,personalselected=5)
#
# # for index in xrange(len(X_train)):
# #     cv2.imshow("x",X_train[index])
# #     # print y_train[index][1],y_train[index][0]
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
#
# trainHog = []
# testHog = []
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
#
#
# for indexOfimage in xrange(len(X_train)):
#     image_Gray, hog = IMA.convertToHOG(X_train[indexOfimage], color=False)
#     # print hog
#     trainHog.append(hog.tolist())
#     # plt.plot(trainHog[0])
#     # plt.show()
#     print indexOfimage
# for indexOfimage in xrange(len(X_test)):
#     image_Gray, hog = IMA.convertToHOG(X_test[indexOfimage], color=False)
#     testHog.append(hog.tolist())
#     print indexOfimage
# # coqnvert class vectors to binary class matrices
#
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# # print Y_train
# Y_test = np_utils.to_categorical(y_test, nb_classes)
#
# trainHog = np.asarray(trainHog)
# trainHog = trainHog.astype('float32')
#
# testHog = np.asarray(testHog)
# testHog = testHog.astype('float32')
# np.save("TrainHog",trainHog)
# np.save("TestHog",testHog)
# np.save("Y_train",Y_train)
# np.save("Y_test",Y_test)
# print "Save...."

trainHog = np.load("TrainHog.npy")
testHog = np.load("TestHog.npy")
Y_train = np.load("Y_train.npy")
Y_test = np.load("Y_test.npy")
print "Loaded.."

inputdimension = trainHog[0].shape[0]

print "x",inputdimension
model = Sequential()
model.add(Dense(2000, input_shape=(inputdimension,)))
model.add(Activation("relu"))
model.add(Dropout(0.1))
for i in xrange(10):
    model.add(Dense(2000))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(trainHog, Y_train, batch_size=batch_size,shuffle=True,
                    nb_epoch=nb_epoch, verbose=1,validation_data=(testHog, Y_test))
score = model.evaluate(testHog, Y_test, verbose=1)


model_json = model.to_json()
with open("model_MLP.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_MLP.h5")
print("Saved model to disk")

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