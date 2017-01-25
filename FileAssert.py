import cv2
import numpy as np
import time
import math
import sys
from os import walk

class FileAssert(object):
    def __init__(self,currentPath = "image"):
        self.currentPath = currentPath
        self.configFile = "Dataset.txt"
        #self.writeEmptyDataset()
        #self.readFile(self.currentPath)

    def writeEmptyDataset(self):
        text_file = open( self.configFile, "w")
        text_file.close()

    def writeFile(self,text):
        text_file = open( self.configFile, "a")
        text_file.write("%s \n" % text)
        text_file.close()

    def readDataSetFile(self):
        with open(self.configFile) as f:
            rawPath  = f.read().splitlines()
        return rawPath

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

# FileAssert().writeDataSetFile(".\Image")