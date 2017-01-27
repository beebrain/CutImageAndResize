import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import time
import math
import sys
from os import walk
import gc
import os
import MainCut as MC

class FileAssert(object):
    def __init__(self,currentPath = "image"):
        self.currentPath = currentPath
        self.configFile = "Dataset.txt"
        self.imageData = "ImageDataset"
        self.detailData = "DetailDataset"
        self.size = 64
        #self.writeEmptyDataset()
        #self.readFile(self.currentPath)

    def writeEmptyDataset(self):
        text_file = open( self.configFile, "w")
        text_file.close()

    def writeFile(self,text):
        text_file = open( self.configFile, "a")
        text_file.write("%s \n" % text)
        text_file.close()

    # for read Dataset.txt is config path of image
    def readDataSetConfigFile(self):
        with open(self.configFile) as f:
            rawPath  = f.read().splitlines()
        return rawPath



    #for read raw dataset , it's have (class,personal,filename,height,width,chanels,imagedata_on_one_dimetion)
    #and unpackdataset to two value one is datadetail ,anather one is imagedataset
    def unpackDataset(self,color=True):
        if not (os.path.exists(self.detailData) and os.path.isfile(self.imageData)):
            detailDataset = np.load(self.detailData + ".npy")
            if color :
                imgDataset = np.load(self.imageData+".npy")
                imgDataset = imgDataset.reshape((-1,self.size,self.size,3))
            else:   # GrayScale
                imgDataset = np.load(self.imageData + "_Gray.npy")
                imgDataset = imgDataset.reshape((-1, self.size, self.size))

            #Make sure dataset print out
            # for index in xrange(len(imgDataset)):
            #     cv2.imshow("x",imgDataset[index])
            #     print detailDataset[index][1],detailDataset[index][0]
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()


            # change Personal to index personal
            detailDataset[:,1] = [str.replace(w,"P","") for w in detailDataset[:,1]]
            # print detailDataset
        return (imgDataset.astype('uint8'),detailDataset[:,:2].astype('int'))

    def resizeImage(self):
        configpath = self.readDataSetConfigFile()
        print configpath
        for listpath in xrange(len(configpath)):
            e = configpath[listpath]
            list = str.split(e, ",")
            image2 = cv2.imread(list[0])
            image2 = cv2.resize(image2, (0, 0), fx=0.3, fy=0.3)
            print list[0]
            imageresize = MC.process(image2,self.size)
            cv2.imwrite(str("./resize/" + list[3] + "_" + list[2] + "_" + str(listpath).zfill(4) + ".jpg"), imageresize)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    def packData(self,path="./resize"):
        if path == "":
            print "Don't have File"
        else:
            imageData = []
            imageData_gray =[]
            detailData = []

            for (dirpath, dirnames, filenames) in walk(path):
                listFolder = []
                listFile = []
                listFile.extend(filenames)
                listFolder.extend(dirnames)

                for iFile in xrange(len(listFile)):
                    filename = listFile[iFile]
                    detailname = str.split(filename, "_")
                    img = cv2.imread(path + "//" + filename)
                    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    (h, w, c) = img.shape

                    img = img.reshape(-1)
                    img_gray = img_gray.reshape(-1)

                    newrow = np.array([detailname[0], detailname[1], filename])  # class,personal,filename,imageFile
                    newrow = np.hstack((newrow, (h, w, c)))

                    if (len(imageData) == 0):
                        imageData = img
                        imageData_gray = img_gray
                        detailData = newrow
                    else:
                        imageData = np.vstack((imageData, img))
                        imageData_gray = np.vstack((imageData_gray,img_gray))
                        detailData = np.vstack((detailData, newrow))
                    if (iFile % 100 == 0):
                        # gc.collect()
                        print iFile

            print "saving..."
            if (len(imageData) == len(detailData)):
                np.save(self.imageData, imageData)
                np.save(self.imageData+"_gray", imageData_gray)
                np.save(self.detailData, detailData)
            else:
                print len(imageData)
                print len(detailData)
                print "Data Not Matched"
            print "saved"


        return imageData,img_gray,detailData

    def confirmDataset(self):
        listdata = np.load("CompleteDataSet.npy")

    def writeDataSetFile(self,path = ""):
        if path == "":
            print "Don't have File"
        else :
            for (dirpath, dirnames, filenames) in walk(path):
                listFolder = []
                listFile = []
                listFile.extend(filenames)
                listFolder.extend(dirnames)
                if len(listFile) != 0 :
                    for iFile in xrange(len(listFile)):
                        self.writeFile(dirpath+"\\"+filenames[iFile]+","+str(str.split(dirpath,"\\")[1:4]).strip('[]').replace("'", "").replace(" ","")+","+filenames[iFile])
                        print str(str.split(dirpath,"\\")[1:4]).strip('[]')+","+filenames[iFile]
                        print dirpath+"\\"+filenames[iFile]+","+str(str.split(dirpath,"\\")[1:4]).strip('[]').replace("'", "").replace(" ","")+","+filenames[iFile]



# FileAssert().resizeImage()
# imageDataset,imageGray,detailDataset = FileAssert().packData()
#
# (x,y)=FileAssert().unpackDataset(False)
#
# class_label = y[:,0]
# personal = y[:,1]
# print len(x)
# rangeindex  = np.where(personal==9)
# class_label_selected = class_label[rangeindex]
# print class_label_selected
# for idx, val in enumerate(rangeindex):
#     print val
    # print "aa",index
    # # cv2.imshow("xxx", x[index])
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # print class_label[index]
    # print personal[index]
