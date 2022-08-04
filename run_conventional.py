# import torch
from torch.utils.data import DataLoader

import logging
import os

from config import parse_args
from utils.data_loaders import *
from utils.log_helpers import *
from utils.helpers import *
from utils.data_transformer import *

######### Configuration #########
######### Configuration #########
######### Configuration #########
args = parse_args()

# Session Parameters
NUM_WORKERS = args.num_workers

# Directory Parameters
DATA_DIR = args.data_dir
DATASET = args.test_dataset

DO_JPEGXL = True
DO_FLIF = True

# Set up logger
filename = 'logs_conventional.txt'
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

logging.info("="*100)
logging.info('DATASET : %s' % (DATASET))

######### Configuration #########
######### Configuration #########
######### Configuration #########

imgs = []

for file in os.listdir(os.path.join(DATA_DIR, DATASET, 'test')):
    if file.endswith('.png'):
        imgs.append(os.path.join(DATA_DIR, DATASET, 'test', file))

imgs = sorted(imgs)


if DO_JPEGXL:
    avg_bpp = 0.0
    avg_time = 0.0

    for imgname in imgs:

        tempname = 'temp.png'
        
        img = cv2.cvtColor(cv2.imread(imgname),cv2.COLOR_BGR2RGB)

        raw_img = doCFA(img)

        h, w = raw_img.shape

        cv2.imwrite(tempname, raw_img.astype(np.uint8))

        start = time()
        os.system('jpegxl/build/tools/cjxl "%s" output.jxl -q 100' % (tempname))
        end = time()        

        filesize = os.stat('output.jxl').st_size

        bpp = 8*filesize / (h*w)
        avg_bpp += bpp
        avg_time += (end - start)

        logging.info("%s : %.4f" % (imgname, bpp))

        os.system('rm "%s"' % (tempname))
        os.system('rm %s' % ('output.jxl'))

    avg_bpp /= len(imgs)
    avg_time /= len(imgs)

    logging.info('JPEGXL Avg BPP : %.4f    Avg Time : %.4f' % (avg_bpp, avg_time))


if DO_FLIF:
    avg_bpp = 0.0
    avg_time = 0.0

    for imgname in imgs:

        tempname = 'temp.png'

        img = cv2.cvtColor(cv2.imread(imgname),cv2.COLOR_BGR2RGB)

        raw_img = doCFA(img)

        h, w = raw_img.shape

        cv2.imwrite(tempname, raw_img.astype(np.uint8))

        start = time()
        os.system('flif -e -Q100 "%s" output.flif' % (tempname))
        end = time()        

        filesize = os.stat('output.flif').st_size

        bpp = 8*filesize / (h*w)
        avg_bpp += bpp
        avg_time += (end - start)

        logging.info("%s : %.4f" % (imgname, bpp))

        os.system('rm "%s"' % (tempname))
        os.system('rm %s' % ('output.flif'))

    avg_bpp /= len(imgs)
    avg_time /= len(imgs)

    logging.info('FLIF Avg BPP : %.4f    Avg Time : %.4f' % (avg_bpp, avg_time))