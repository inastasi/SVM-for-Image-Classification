# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:23:09 2019

@author: marco
"""

"""
this is the code you need to run to import the data. 
You just have to change line 40 putting the correct path.
"""

import numpy as np

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def returnData():

    X_all_labels, y_all_labels = load_mnist('..\Data', kind='train')


    """
    We are only interested in the items with label 2, 4 and 6.
    Only a subset of 1000 samples per class will be used.
    """
    indexLabel2 = np.where((y_all_labels==2))
    xLabel2 =  X_all_labels[indexLabel2][:1000,:].astype('float64')
    yLabel2 = y_all_labels[indexLabel2][:1000].astype('float64')

    indexLabel4 = np.where((y_all_labels==4))
    xLabel4 =  X_all_labels[indexLabel4][:1000,:].astype('float64')
    yLabel4 = y_all_labels[indexLabel4][:1000].astype('float64')

    indexLabel6 = np.where((y_all_labels==6))
    xLabel6 =  X_all_labels[indexLabel6][:1000,:].astype('float64')
    yLabel6 = y_all_labels[indexLabel6][:1000].astype('float64')

    return xLabel2,yLabel2,xLabel4,yLabel4,xLabel6,yLabel6



"""
To train a SVM, you have to convert the labels of the two classes of interest into '+1' and '-1'.
"""
