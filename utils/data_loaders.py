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

class DivideCollate(object):

    def __init__(self):
        
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    
    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        raw_imgs, img_names, paddings = zip(*batch)

        img_a_out, img_b_out, img_c_out, img_d_out, raw_img_out, img_name_out, padding_out = [], [], [], [], [], [], []

        for raw_img, img_name, padding in zip(raw_imgs, img_names, paddings):

            r_img = raw_img[0::2, 0::2]
            g1_img = raw_img[0::2, 1::2]
            g2_img = raw_img[1::2, 0::2]
            b_img = raw_img[1::2, 1::2]

            rggb_img = np.stack((r_img, g1_img, g2_img, b_img), axis=-1)

            yuvd_img = RGGB2YUVD(rggb_img)

            yuvd_img = self.transform(yuvd_img)
            raw_img = self.transform(raw_img)

            yuvd_img = space_to_depth_tensor(yuvd_img)

            img_a, img_b, img_c, img_d = yuvd_img[:,0], yuvd_img[:,1], yuvd_img[:,2], yuvd_img[:,3]

            img_a_out.append(img_a)
            img_b_out.append(img_b)
            img_c_out.append(img_c)
            img_d_out.append(img_d)
            raw_img_out.append(raw_img)
            img_name_out.append(img_name)
            padding_out.append(padding)
        
        img_a_out = torch.cat([t.unsqueeze(0) for t in img_a_out], 0)
        img_b_out = torch.cat([t.unsqueeze(0) for t in img_b_out], 0)
        img_c_out = torch.cat([t.unsqueeze(0) for t in img_c_out], 0)
        img_d_out = torch.cat([t.unsqueeze(0) for t in img_d_out], 0)
        raw_out = torch.cat([t.unsqueeze(0) for t in raw_img_out], 0)

        return img_a_out, img_b_out, img_c_out, img_d_out, raw_out, img_name_out, padding_out

# Synthesized Dataset : DIV2K, CLIC.m, CLIC.p, Kodak, FLICKR2K, etc
class Synthesized_Dataset(torch.utils.data.Dataset):
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
        img = cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB)

        # Crop image for train dataset
        if self.subset == 'train':
            img = RandomCrop(img, 2*CROP_SIZE)
        img, padding = pad_img(img)

        raw_img = doCFA(img)

        return raw_img.astype(np.float32), self.imgs[idx], padding
    
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    args = parse_args()

    PRINT_NUM = 10

    subset = 'train'
    if subset=='train':
        BATCH_SIZE = 4
        DATASET = args.train_dataset
    else:
        BATCH_SIZE = 1
        DATASET = args.test_dataset

    # Create directory to save images
    PLAYGROUND_DIR = 'playground/'

    if not os.path.exists(PLAYGROUND_DIR):
        os.mkdir(PLAYGROUND_DIR)

    # Dataloader
    dataset = Synthesized_Dataset(args, subset)

    # Collate
    Collate_ = DivideCollate()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=Collate_)

    for i, data in enumerate(dataloader, 0):

        img_a, img_b, img_c, img_d, rggb_img, img_name, padding = data

        BATCH_SIZE = list(img_a.shape)[0]

        for b_idx in range(BATCH_SIZE):
            c_img_a, c_img_b, c_img_c, c_img_d, c_rggb_img, c_img_names, c_padding = img_a[b_idx].permute(1,2,0).numpy(), img_b[b_idx].permute(1,2,0).numpy(), img_c[b_idx].permute(1,2,0).numpy(), img_d[b_idx].permute(1,2,0).numpy(), rggb_img[b_idx].permute(1,2,0).numpy(), img_name[b_idx], padding[b_idx]

            c_rggb_img = c_rggb_img[:,:,0].astype(np.uint8)

            recon_img = np.stack((c_img_a, c_img_b, c_img_c, c_img_d), axis=0)
            recon_img = depth_to_space(recon_img)
            recon_img = YUVD2RAW(recon_img)
            recon_img = recon_img.astype(np.uint8)

            diff = np.sum(np.sum(abs(c_rggb_img - recon_img)))
            print(diff)

        if i == PRINT_NUM:
            sys.exit(1)