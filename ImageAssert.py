import cv2
import skimage.exposure as skiexposure
import skimage.feature as skfeature
import numpy as np
import matplotlib.pyplot as plt
import time

defualtThreshold = 150

def convertToHOG(imageData,color=True):
    startTime = time.time()
    #resize image
    # cv2.imshow("GrayImage", imageData)
    # cv2.waitKey(0)
    if(color == True):
        image_Gray = cv2.cvtColor(imageData, cv2.COLOR_RGB2GRAY)
    else:
        image_Gray = imageData

    # hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)


    #Hog Convert
    fd, hog_image = skfeature.hog(image_Gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)


    #Rescale histogram for better display

    hog_image_rescaled = skiexposure.histogram(hog_image, nbins=100)
    # cv2.imshow("HOG",hog_image)

    #plot HOG
    # plt.plot(fd)

    #print("process Time %s seconds ---" % (time.time() - startTime))
    #cv2.waitKey(0)
    return image_Gray,fd

#convertToHOG("image/A01/23.0.jpg")