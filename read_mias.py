import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

def read_mias(path):
    ls = os.listdir('./' + path + '/')
    stack = np.zeros((len(ls), 1024, 1024), dtype = np.uint8)
    for i,f in enumerate(ls):
        #print(os.getcwd() +'/'+ path +'/'+ f)
        with open(os.getcwd() +'/'+ path +'/'+ f, 'rb') as pgmf:
            im = plt.imread(pgmf)
        im = np.array(im, dtype = np.uint8)
        stack[i] = im

    return stack
