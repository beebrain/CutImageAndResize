# Create Image Resize ROI this file Call MainCUt
# It read File image from Dataset that create by FileAssert and send Image File to MainCut and get ROI back from Maincut
# Create Config file Before run this file

import cv2
import numpy as np
import math
import os
import MainCut as MC
import FileAssert as FA
import sys
from os import walk

def resizeImage():
    fa = FA.FileAssert()
    configpath = fa.readDataSetConfigFile()
    print configpath
    for listpath in xrange(len(configpath)):
        e = configpath[listpath]
        list = str.split(e,",")
        image2 = cv2.imread(list[0])
        image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)
        print list[0]
        imageresize = MC.process(image2)
        cv2.imwrite(str("./resize/"+list[3]+"_"+list[2]+"_"+str(listpath).zfill(4)+".jpg"),imageresize)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
