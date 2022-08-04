import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append('.')
from config import parse_args
from utils.data_transformer import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, downscale_ratio):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.imgs = []
        self.subset = subset
        self.downscale_ratio = downscale_ratio

        if subset == 'train':
            self.dataset = args.train_dataset
        elif subset == 'test':
            self.dataset = args.test_dataset

        for file in os.listdir(os.path.join(args.data_dir, self.dataset, subset)):
            if file.endswith('.png'):
                self.imgs.append(file)

        self.imgs = sorted(self.imgs)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, idx):

        DATA_DIR = self.args.data_dir
        DATASET = self.dataset
        CROP_SIZE = self.args.crop_size

        if self.downscale_ratio != 1:
            DATASET = DATASET.replace('/', '') + '_down' + str(self.downscale_ratio)

        # Read in Image
        imgpath = os.path.join(DATA_DIR, DATASET, self.subset, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB)

        # Crop image for train dataset
        if self.subset == 'train':
            img = RandomCrop(img, CROP_SIZE)

        ori_img = img

        img = RGB2YUV(img)
        img = self.transform(img)

        img, padding = pad_img(img)
        img = space_to_depth_tensor(img)

        img_a, img_b, img_c, img_d = img[:,0], img[:,1], img[:,2], img[:,3]

        return img_a, img_b, img_c, img_d, ori_img, imgpath, padding, self.downscale_ratio
    
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    args = parse_args()

    DOWNSCALE_RATIO = 2

    subset = 'test'
    if subset=='train':
        BATCH_SIZE = args.batch_size
    else:
        BATCH_SIZE = 1

    # Create directory to save images
    PLAYGROUND_DIR = 'playground/'

    if not os.path.exists(PLAYGROUND_DIR):
        os.mkdir(PLAYGROUND_DIR)

    dataset = Dataset(args, subset, DOWNSCALE_RATIO)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader, 0):

        img_a, img_b, img_c, img_d, ori_imgs, img_names, padding, downscale_ratio = data

        B, _, H, W, = img_a.shape

        img_a = torch.unsqueeze(img_a, dim=2)

        img_a = (img_a.squeeze()).permute(1,2,0)
        img_a = img_a.numpy()
        img_a = YUV2RGB(img_a)

        img_name = PLAYGROUND_DIR + 'div2k/' + img_names[0]
        cv2.imwrite(img_name, cv2.cvtColor(img_a,cv2.COLOR_RGB2BGR))