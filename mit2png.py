import cv2
import numpy as np
from process_raw import DngFile
from tqdm import tqdm
import os
import rawpy
import sys

DATA_DIR = '../DATASET/Compression'
DATASET = 'mit_dng'
SUBSET = 'train'
SAVE_DIR = os.path.join(DATA_DIR, 'mit16')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, SUBSET), exist_ok=True)


img_list = []

for file in os.listdir(os.path.join(DATA_DIR, DATASET, SUBSET)):
    if file.endswith('.dng'):
        img_list.append(file)

img_list = sorted(img_list)

img_idx = 1

for imgname in tqdm(img_list):
    with rawpy.imread(os.path.join(DATA_DIR, DATASET, SUBSET, imgname)) as img:
        if img.raw_pattern is not None:
            with DngFile.read(os.path.join(DATA_DIR, DATASET, SUBSET, imgname)) as dng:
                raw = dng.raw_image
                pattern = dng.pattern

                H, W = raw.shape

                raw = raw[:H-(H%4), :W-(W%4)]

                if pattern == 'RGGB':
                    bayerpattern = 'R_G1_G2_B'
                elif pattern == 'GRBG':
                    bayerpattern = 'G1_R_B_G2'
                elif pattern == 'BGGR':
                    bayerpattern = 'B_G2_G1_R'
                elif pattern == 'GBRG':
                    bayerpattern = 'G2_B_R_G1'
                else:
                    continue

                bayerpattern = bayerpattern.split('_')

                CFA_img = {}
                CFA_img[bayerpattern[0]] = raw[0::2, 0::2]
                CFA_img[bayerpattern[1]] = raw[0::2, 1::2]
                CFA_img[bayerpattern[2]] = raw[1::2, 0::2]
                CFA_img[bayerpattern[3]] = raw[1::2, 1::2]

                rggb_img = np.stack((CFA_img['R'], CFA_img['G1'], CFA_img['G2'], CFA_img['B']), axis=-1)

                H,W,_ = rggb_img.shape

                raw_img = np.zeros((2*H, 2*W))

                raw_img[0::2,0::2] = rggb_img[:,:,0]
                raw_img[0::2,1::2] = rggb_img[:,:,1]
                raw_img[1::2,0::2] = rggb_img[:,:,2]
                raw_img[1::2,1::2] = rggb_img[:,:,3]

                raw_img = raw_img.astype(np.uint16)

                outname = os.path.join(SAVE_DIR, SUBSET, str(img_idx).zfill(4) + '.png')
                cv2.imwrite(outname, raw_img)
                
                img_idx +=1