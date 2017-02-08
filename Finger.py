import numpy as np
import os
import FileAssert as fa
import cv2


def load_data(color = True,personalselected = 1):
    fileAssert = fa.FileAssert(size=64)
    xdata,x_gData,ydata = fileAssert.unpackDataset()    #color,gray,detail
    #detail description
    #[class,personal,filename]


    class_label = ydata[:,0]
    personal = ydata[:,1]
    trainingindex  = np.where(personal <> personalselected)     #select dataset
    testingindex = np.where(personal == personalselected)  # select dataset

    print trainingindex
    print testingindex
    if color == True:
        X_train = xdata[trainingindex]
        X_test = xdata[testingindex]
    else:
        X_train = x_gData[trainingindex]
        X_test = x_gData[testingindex]


    Y_train = class_label[trainingindex]
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


# (X_train, y_train), (X_test, y_test) = load_data(False)
# print X_test