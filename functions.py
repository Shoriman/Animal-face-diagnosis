import os
import glob

import cv2
import numpy as np
from sklearn import cluster

def get_dataset(dataset_path):
    # Remove hidden files
    dir_list = os.listdir(dataset_path)
    for dir_name in dir_list:
        if ('.' in dir_name):
            dir_list.remove(dir_name)
    dir_list.sort()

    # Get images and corresponding class labels
    images = []
    labels = []
    for dir_name in dir_list:
        img_list = glob.glob(os.path.join(dataset_path, dir_name, '*.tif'))
        img_list.sort()
        for img_path in img_list:
            images.append(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
            labels.append(dir_name)

    return images, labels

def calc_VW(images, descriptor, n_clusters):
    # Extract features from images
    features = []
    for img in images:
        features.extend(descriptor.detectAndCompute(img,None)[1])
    # K-means Clustering
    VW = cluster.MiniBatchKMeans(n_clusters=n_clusters).fit(features).cluster_centers_

    return VW

def calc_hist(img, descriptor, VW):
    # Extract features from an image
    features = descriptor.detectAndCompute(img,None)[1]
    # Calculate histogram based on VW
    hist = np.zeros(len(VW))
    for f in features:
        hist[((VW-f)**2).sum(axis=1).argmin()] += 1

    return hist

    
    
    
    
    
    
