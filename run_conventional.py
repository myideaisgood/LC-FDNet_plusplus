# import torch
from torch.utils.data import DataLoader

import logging
import os

from config import parse_args
from utils.average_meter import AverageMeter
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
DATASET = args.test_dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)

# Check if directory does not exist
create_path('experiments/')
create_path(EXP_DIR)
create_path(CKPT_DIR)
create_path(LOG_DIR)
create_path(os.path.join(LOG_DIR, 'train'))
create_path(os.path.join(LOG_DIR, 'test'))

# Set up logger
filename = os.path.join(LOG_DIR, 'logs_conventional.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

# Set up Dataset
test_dataset = Dataset(args, 'test', 1)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=2,
    shuffle=False
)


# Run jpeg-xl
jpegxl_avg_bpp = 0.0
jpegxl_avg_time = 0.0

for idx, data in enumerate(test_dataloader):

    _, _, _, _, ori_img, img_name, padding, _ = data

    ori_img = ori_img[0].numpy().astype(np.uint8)

    h, w, _ = ori_img.shape

    savename = 'temp.png'

    cv2.imwrite(savename, cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR))

    start = time()
    os.system('jpegxl/build/tools/cjxl "%s" output.jxl -q 100' % (savename))
    end = time()

    filesize = os.stat('output.jxl').st_size

    bpp = 8*filesize / (h*w)
    jpegxl_avg_bpp += bpp
    jpegxl_avg_time += (end - start)

    logging.info("%s : %.4f" % (img_name, bpp))

    os.system('rm "%s"' % (savename))
    os.system('rm %s' % ('output.jxl'))

jpegxl_avg_bpp /= len(test_dataloader)
jpegxl_avg_time /= len(test_dataloader)

logging.info('JPEGXL Avg BPP : %.4f    Avg Time : %.4f' % (jpegxl_avg_bpp, jpegxl_avg_time))


# Run flif
flif_avg_bpp = 0.0
flif_avg_time = 0.0

for idx, data in enumerate(test_dataloader):

    _, _, _, _, ori_img, img_name, padding, _ = data

    ori_img = ori_img[0].numpy().astype(np.uint8)

    h, w, _ = ori_img.shape

    savename = 'temp.png'

    cv2.imwrite(savename, cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR))

    start = time()
    os.system('flif -e -Q100 "%s" output.flif' % (savename))
    end = time()

    filesize = os.stat('output.flif').st_size

    bpp = 8*filesize / (h*w)
    flif_avg_bpp += bpp
    flif_avg_time += (end - start)

    logging.info("%s : %.4f" % (img_name, bpp))

    os.system('rm "%s"' % (savename))
    os.system('rm %s' % ('output.flif'))

flif_avg_bpp /= len(test_dataloader)
flif_avg_time /= len(test_dataloader)

logging.info('FLIF Avg BPP : %.4f    Avg Time : %.4f' % (flif_avg_bpp, flif_avg_time))