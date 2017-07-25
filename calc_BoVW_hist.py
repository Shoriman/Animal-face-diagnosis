#!/usr/bin/env python2.7
import os
import argparse
import pickle

import cv2

from functions import *

def main():
    # Feature descriptor list
    descriptors = {
            'KAZE':cv2.KAZE_create
            }
    
    # Process command-line argument
    parser = argparse.ArgumentParser(description='Calculate BoVW(VW) and histograms of training images')
    parser.add_argument('--train_path','-t',default='animal_faces_train',help='Path of train images')
    parser.add_argument('--vw_path','-v',default='animal_face_vw',help='Path of calculated vw of training images')
    parser.add_argument('--hist_path','-o',default='animal_face_histgrams',help='Path of calculated histgrams of training images')
    parser.add_argument('--n_clusters','-n',type=int,default=128,help='Number of clusters in k-means clustering')
    parser.add_argument('--descriptor','-d',default='KAZE', help='Type of feature desciptor')
    parser.add_argument('--threshold','-th',type=int,default=0,help='Threshold of detector')
    args = parser.parse_args()

    # Get train dataset
    train_images, train_labels = get_dataset(args.train_path)

    # Calculate features and VW
    descriptor = descriptors[args.descriptor](args.threshold)
    VW = calc_VW(train_images,descriptor,args.n_clusters)

    # Calculate VW histgrams of training images
    hist_list = []
    for index, img in enumerate(train_images):
        hist_list.append((calc_hist(img,descriptor,VW),train_labels[index]))

    # Save VW and histgrams of training images
    with open(args.vw_path,'wb') as f:
        pickle.dump(VW,f)
        f.close()
    with open(args.hist_path,'wb') as f:
        pickle.dump(hist_list,f)
        f.close()

if __name__=='__main__':
    main()
    
    
    
    
