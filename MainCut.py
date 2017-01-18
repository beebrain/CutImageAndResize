import cv2
import numpy as np
import math

def thresholdWithYCC(img):
    # convert to hsv value for extract skin tone
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # lower = np.array([0, 133, 77])
    # upper = np.array([255, 173, 127])
    lower = np.array([0, 120, 60])
    upper = np.array([255, 180, 130])
    skintone = cv2.inRange(ycc, lower, upper)

    skintone_mask = cv2.bitwise_and(img, img, mask=skintone)  # extract skin tone
    cv2.imshow("skintone_mask",skintone_mask)

    # convert to gray scale for extract background and Light luminate
    gray = cv2.cvtColor(skintone_mask, cv2.COLOR_BGR2GRAY)      #convert to gray scale
    valueofGaussian = (13, 13)
    gray = cv2.GaussianBlur(gray, valueofGaussian, 0)           # used GaussianBlur for smote edge
    cv2.imshow("gray_mask", gray)
    res, thres = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY )     #Extract hand from backgroud + cv2.THRESH_OTSU
    cv2.imshow("GraySkintone", thres)

    mask_skinTone_binary = fillHold(thres)
    real_skinTone = cv2.bitwise_and(img, img, mask=mask_skinTone_binary)
    cv2.imshow("real_skinTone", real_skinTone)
    return mask_skinTone_binary,real_skinTone

def fillHold(img):
    im_floodfill = img.copy()
    h,w = img.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(im_floodfill,mask,(0,0),255)      # Fill with white screen
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)    # invert black to white for mask
    im_out = img|im_floodfill_inv

    #cv2.imshow(" Image", img)
    #cv2.imshow("im_floodfill Image", im_floodfill)
    #cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    #cv2.imshow("Foreground", im_out)

    return im_out

def process(img):
    thresh_skin,skincolor = thresholdWithYCC(img)
    cv2.imshow("skinColor",skincolor)

    contours , hierarchy = cv2.findContours(thresh_skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Contour = np.zeros(img.shape,dtype= np.uint8)
    cv2.drawContours(Contour, contours, -1, (255, 255, 255), 1)          # drawContour we can't use grayscale image

    Contour  = cv2.cvtColor(Contour,cv2.COLOR_RGB2GRAY)                 # convert to 2 dimentions
    ret, Contour = cv2.threshold(Contour, 254, 255, cv2.THRESH_BINARY)  # make sure only 255 value in array
    cv2.imshow("Contour",Contour)

    imageGray = cv2.cvtColor(skincolor, cv2.COLOR_BGR2GRAY)             # Skincolor ROT Convert TO Gray Scale
    cv2.imshow("gray",imageGray)

image2 = cv2.imread("Image/P1130707.jpg")
small = cv2.resize(image2, (0,0), fx=0.1, fy=0.1)
process(small)
cv2.waitKey(0)
cv2.destroyAllWindows()


#
#
# # select Crop Image
# # cv2.rectangle(image2, (100, 100), (435, 435), (0, 0, 255), 2)
# cv2.imshow("ColorImage", image2)
#
# # Crop from x, y, w, h -> 100, 200, 300, 400
# # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
# crop_img = image2
# cv2.imshow("Crop_image", crop_img)
# # cv2.rectangle(image2, (100, 100), (435, 435), (0, 0, 255), 2)
#
# image = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
# # hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
#
#
# ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
# # mask = cv2.inRange(hsv, lower_blue, upper_blue)
# # res = cv2.bitwise_and(crop_img,crop_img, mask= mask)
#
# im_floodfill = thresh.copy()
# cv2.imshow("Threshold", thresh)
# # convert mask
#
# h, w = thresh.shape[:2]
# mask = np.zeros((h + 2, w + 2), np.uint8)
#
# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0, 0), 255);
# im_floodfill_inv = cv2.bitwise_not(im_floodfill)
# im_out = thresh | im_floodfill_inv
#
# fd, hog_image = skfeature.hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
#
# # Rescale histogram for better display
# hog_image_rescaled = skiexposure.histogram(hog_image, nbins=100)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(image, contours, -1, (0, 250, 250), 2)
#
# cv2.imshow("contours", im_out)
# cv2.imshow("HOG_Output", hog_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
