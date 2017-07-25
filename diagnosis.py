#!/usr/bin/env python2.7
import os,sys
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
    parser = argparse.ArgumentParser(description='Execute animal-face diagnosis')
    parser.add_argument('input_img',help='Path of subject image')
    parser.add_argument('--cascade_path','-c',default='haarcascade_frontalface_alt.xml',help='Path of trained haarcascade')
    parser.add_argument('--vw_path','-v',default='animal_face_vw',help='Path of reference BoVW of training images')
    parser.add_argument('--hist_path','-r',default='animal_face_histgrams',help='Path of reference histgrams of training images')
    parser.add_argument('--descriptor','-d',default='KAZE', help='Type of feature desciptor')
    parser.add_argument('--threshold','-th',type=int,default=0,help='Threshold of detector')
    parser.add_argument('--n_img','-n',type=int,default=4,help='Number of images per class')
    parser.add_argument('--is_face','-f',type=int,default=1,help='Whether or not to extract face area')
    args = parser.parse_args()

    # Read input image
    sub_img = cv2.imread(args.input_img,0)

    # Extract face area
    if args.is_face:
        cascade = cv2.CascadeClassifier(args.cascade_path)
        facerect = cascade.detectMultiScale(sub_img,scaleFactor=1.1,minNeighbors=1,minSize=(1,1))
        if len(facerect) == 1:
            facerect = facerect[0]
            face_img = sub_img[facerect[1]:facerect[1]+facerect[3],facerect[0]:facerect[0]+facerect[2]]
        else:
            print 'failed to detect one face area!!'
            sys.exit()
    else:
        face_img = sub_img

    # Calculate features of face image
    with open(args.vw_path,'rb') as f:
        ref_VW = pickle.load(f)
        f.close()
    descriptor = descriptors[args.descriptor](threshold=args.threshold)
    face_hist = calc_hist(face_img,descriptor,ref_VW)

    # Calculate similarity of histogram with animals'
    with open(args.hist_path,'rb') as f:
        ref_hist_list = pickle.load(f)
        f.close()
    results = []
    for i in xrange(0,len(ref_hist_list),args.n_img):
        target_hist_list = ref_hist_list[i:i+args.n_img]
        similarity = calc_hist_intersection(face_hist,target_hist_list)
        results.append((similarity,target_hist_list[0][1]))

    # Show top-1
    results.sort(reverse=True)
    print 'Your face looks like '+results[0][1]+'!'

if __name__=='__main__':
    main()
    

    
