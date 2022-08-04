import cv2
import os
import numpy as np
from process_raw import DngFile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append('.')
from utils.data_transformer import *
from config import parse_args

def raw2data(raw_img):

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    r_img = raw_img[0::2, 0::2]
    g1_img = raw_img[0::2, 1::2]
    g2_img = raw_img[1::2, 0::2]
    b_img = raw_img[1::2, 1::2]

    rggb_img = np.stack((r_img, g1_img, g2_img, b_img), axis=-1)

    yuvd_img = RGGB2YUVD(rggb_img)

    yuvd_img = transform(yuvd_img)
    raw_img = transform(raw_img)

    yuvd_img = space_to_depth_tensor(yuvd_img)

    img_a, img_b, img_c, img_d = yuvd_img[:,0], yuvd_img[:,1], yuvd_img[:,2], yuvd_img[:,3]

    return img_a, img_b, img_c, img_d

class DivideCollate(object):

    def __init__(self):
        
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    
    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        raw_imgs_msb, raw_imgs_lsb, img_names, paddings = zip(*batch)

        img_a_msb_out, img_b_msb_out, img_c_msb_out, img_d_msb_out = [], [], [], []
        img_a_lsb_out, img_b_lsb_out, img_c_lsb_out, img_d_lsb_out = [], [], [], []

        raw_img_out, img_name_out, padding_out = [], [], []

        for raw_img_msb, raw_img_lsb, img_name, padding in zip(raw_imgs_msb, raw_imgs_lsb, img_names, paddings):

            img_a_msb, img_b_msb, img_c_msb, img_d_msb = raw2data(raw_img_msb)
            img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb = raw2data(raw_img_lsb)

            img_a_msb_out.append(img_a_msb)
            img_b_msb_out.append(img_b_msb)
            img_c_msb_out.append(img_c_msb)
            img_d_msb_out.append(img_d_msb)

            img_a_lsb_out.append(img_a_lsb)
            img_b_lsb_out.append(img_b_lsb)
            img_c_lsb_out.append(img_c_lsb)
            img_d_lsb_out.append(img_d_lsb)

            raw_img = 256*raw_img_msb + raw_img_lsb
            raw_img = self.transform(raw_img)

            raw_img_out.append(raw_img)
            img_name_out.append(img_name)
            padding_out.append(padding)
        
        img_a_msb_out = torch.cat([t.unsqueeze(0) for t in img_a_msb_out], 0)
        img_b_msb_out = torch.cat([t.unsqueeze(0) for t in img_b_msb_out], 0)
        img_c_msb_out = torch.cat([t.unsqueeze(0) for t in img_c_msb_out], 0)
        img_d_msb_out = torch.cat([t.unsqueeze(0) for t in img_d_msb_out], 0)

        img_a_lsb_out = torch.cat([t.unsqueeze(0) for t in img_a_lsb_out], 0)
        img_b_lsb_out = torch.cat([t.unsqueeze(0) for t in img_b_lsb_out], 0)
        img_c_lsb_out = torch.cat([t.unsqueeze(0) for t in img_c_lsb_out], 0)
        img_d_lsb_out = torch.cat([t.unsqueeze(0) for t in img_d_lsb_out], 0)

        raw_out = torch.cat([t.unsqueeze(0) for t in raw_img_out], 0)

        return img_a_msb_out, img_b_msb_out, img_c_msb_out, img_d_msb_out, img_a_lsb_out, img_b_lsb_out, img_c_lsb_out, img_d_lsb_out, raw_out, img_name_out, padding_out

# SIDD Dataset
class SIDD_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'test')

        self.args = args
        self.imgs = []
        self.subset = subset

        if subset == 'train':
            self.dataset = args.train_dataset
        elif subset == 'test':
            self.dataset = args.test_dataset

        for file in os.listdir(os.path.join(args.data_dir, self.dataset, subset)):
            if file.endswith('.png'):
                self.imgs.append(file)

        self.imgs = sorted(self.imgs)

    def __getitem__(self, idx):

        DATA_DIR = self.args.data_dir
        CROP_SIZE = self.args.crop_size

        # Read in Image
        imgpath = os.path.join(DATA_DIR, self.dataset, self.subset, self.imgs[idx])
        img = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH)
        # Image : (H, W, 1)

        if 'GP' in self.imgs[idx]:
            bayerpattern = 'B_G1_G2_R'
        elif 'IP' in self.imgs[idx]:
            bayerpattern = 'R_G1_G2_B'
        elif 'S6' in self.imgs[idx]:
            bayerpattern = 'G1_R_B_G2'
        elif 'N6' in self.imgs[idx]:
            bayerpattern = 'B_G1_G2_R'
        elif 'G4' in self.imgs[idx]:
            bayerpattern = 'B_G1_G2_R'

        bayerpattern = bayerpattern.split('_')

        CFA_img = {}
        CFA_img[bayerpattern[0]] = img[0::2, 0::2]
        CFA_img[bayerpattern[1]] = img[0::2, 1::2]
        CFA_img[bayerpattern[2]] = img[1::2, 0::2]
        CFA_img[bayerpattern[3]] = img[1::2, 1::2]

        # rggb_img : (H, W, 4)
        rggb_img = np.stack((CFA_img['R'], CFA_img['G1'], CFA_img['G2'], CFA_img['B']), axis=-1)

        # Crop image for train dataset
        if self.subset == 'train':
            rggb_img = RandomCrop(rggb_img, CROP_SIZE)
        rggb_img, padding = pad_img(rggb_img)

        H,W,_ = rggb_img.shape

        raw_img = np.zeros((2*H, 2*W))

        raw_img[0::2,0::2] = rggb_img[:,:,0]
        raw_img[0::2,1::2] = rggb_img[:,:,1]
        raw_img[1::2,0::2] = rggb_img[:,:,2]
        raw_img[1::2,1::2] = rggb_img[:,:,3]

        raw_img_msb, raw_img_lsb = divmod(raw_img, 1<<8)

        return raw_img_msb.astype(np.float32), raw_img_lsb.astype(np.float32), self.imgs[idx], padding
    
    def __len__(self):
        return len(self.imgs)


# MIT Dataset
class MIT_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'test')

        self.args = args
        self.imgs = []
        self.subset = subset

        if subset == 'train':
            self.dataset = args.train_dataset
        elif subset == 'test':
            self.dataset = args.test_dataset

        for file in os.listdir(os.path.join(args.data_dir, self.dataset, subset)):
            if file.endswith('.png'):
                self.imgs.append(file)

        self.imgs = sorted(self.imgs)

    def __getitem__(self, idx):

        DATA_DIR = self.args.data_dir
        CROP_SIZE = self.args.crop_size

        # Read in Image
        imgpath = os.path.join(DATA_DIR, self.dataset, self.subset, self.imgs[idx])
        img = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH)

        # Image : (H, W, 1)
        bayerpattern = 'R_G1_G2_B'
        bayerpattern = bayerpattern.split('_')

        CFA_img = {}
        CFA_img[bayerpattern[0]] = img[0::2, 0::2]
        CFA_img[bayerpattern[1]] = img[0::2, 1::2]
        CFA_img[bayerpattern[2]] = img[1::2, 0::2]
        CFA_img[bayerpattern[3]] = img[1::2, 1::2]

        # rggb_img : (H, W, 4)
        rggb_img = np.stack((CFA_img['R'], CFA_img['G1'], CFA_img['G2'], CFA_img['B']), axis=-1)

        # Crop image for train dataset
        if self.subset == 'train':
            rggb_img = RandomCrop(rggb_img, CROP_SIZE)
        rggb_img, padding = pad_img(rggb_img)

        H,W,_ = rggb_img.shape

        raw_img = np.zeros((2*H, 2*W))

        raw_img[0::2,0::2] = rggb_img[:,:,0]
        raw_img[0::2,1::2] = rggb_img[:,:,1]
        raw_img[1::2,0::2] = rggb_img[:,:,2]
        raw_img[1::2,1::2] = rggb_img[:,:,3]

        raw_img_msb, raw_img_lsb = divmod(raw_img, 1<<8)

        return raw_img_msb.astype(np.float32), raw_img_lsb.astype(np.float32), self.imgs[idx], padding
    
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    args = parse_args()

    PRINT_NUM = 10
    DATA_DIR = args.data_dir

    subset = 'test'
    if subset=='train':
        BATCH_SIZE = 1
        DATASET = args.train_dataset
    else:
        BATCH_SIZE = 1
        DATASET = args.test_dataset

    # Create directory to save images
    PLAYGROUND_DIR = 'playground/'

    if not os.path.exists(PLAYGROUND_DIR):
        os.mkdir(PLAYGROUND_DIR)

    # Dataloader
    if 'SIDD' in DATASET:
        dataset = SIDD_Dataset(args, subset)
    elif 'mit' in DATASET:
        dataset = MIT_Dataset(args, subset)

    # Collate
    Collate_ = DivideCollate()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=Collate_)

    for i, data in enumerate(dataloader, 0):

        img_a_msb, img_b_msb, img_c_msb, img_d_msb, img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb, rggb_img, img_name, padding = data

        img_msb = torch.stack((img_a_msb, img_b_msb, img_c_msb, img_d_msb), dim=2)
        img_lsb = torch.stack((img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb), dim=2)

        img_msb = depth_to_space_tensor(img_msb)
        img_lsb = depth_to_space_tensor(img_lsb)

        BATCH_SIZE = list(img_a_msb.shape)[0]

        for b_idx in range(BATCH_SIZE):
            c_rggb_img = rggb_img[b_idx].permute(1,2,0)[:,:,0].numpy()

            c_img_msb, c_img_lsb = img_msb[b_idx].permute(1,2,0).numpy(), img_lsb[b_idx].permute(1,2,0).numpy()
            c_img_msb, c_img_lsb = YUVD2RAW(c_img_msb), YUVD2RAW(c_img_lsb)

            recon_img = 256*c_img_msb + c_img_lsb

            diff = np.sum(np.sum(abs(recon_img - c_rggb_img)))
            print(diff)

        if i == PRINT_NUM:
            sys.exit(1)

