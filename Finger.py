import numpy as np
import os
import FileAssert as fa
import cv2

def load_data(color = True):
    fileAssert = fa.FileAssert()
    xdata,ydata = fileAssert.unpackDataset(color)
    class_label = ydata[:,0]
    personal = ydata[:,1]
    trainingindex  = np.where(personal<=9)     #select dataset
    testingindex = np.where(personal > 9)  # select dataset

    print trainingindex
    class_label_selected = class_label[trainingindex]
    # X_train = np.zeros((trainingindex, 3, 200, 200), dtype="uint8")
    # Y_train = np.zeros((trainingindex,), dtype="uint8")

    X_train = xdata[trainingindex]
    Y_train = class_label[trainingindex]

    X_test = xdata[testingindex]
    Y_test = class_label[testingindex]

    Y_train = np.reshape(Y_train, (len(Y_train), 1))
    Y_test = np.reshape(Y_test, (len(Y_test), 1))

    # rerun category start at 0
    Y_train -= 1
    Y_test -= 1
    # if K.image_dim_ordering() == 'tf':
    #     X_train = X_train.transpose(0, 2, 3, 1)
    #     X_test = X_test.transpose(0, 2, 3, 1)

    return (X_train, Y_train), (X_test, Y_test)


# (X_train, y_train), (X_test, y_test) = load_data()
# print X_test