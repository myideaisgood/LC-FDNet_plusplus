# import torch
from torch.utils.data import DataLoader

import logging
import os
from tqdm import tqdm

from config import parse_args
from eval import evaluate
from utils.average_meter import AverageMeter
from utils.data_loaders import Dataset
from utils.log_helpers import *
from utils.helpers import *
from utils.data_transformer import *

######### Configuration #########
######### Configuration #########
######### Configuration #########
args = parse_args()

# Design Parameters
CROP_SIZE = args.crop_size
HIDDEN_UNIT = args.hidden_unit
LAMBDA_PRED = args.lambda_pred
LAMBDA_EV = args.lambda_ev
LAMBDA_BR = args.lambda_br

# Session Parameters
GPU_NUM = args.gpu_num
NUM_WORKERS = args.num_workers
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
LR = args.lr
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

DO_EVAL = args.do_eval
EVAL_ALL = args.eval_all
SAVE_EVERY = args.save_every
PRINT_EVERY = args.print_every
EVAL_EVERY = args.eval_every

# Directory Parameters
TRAIN_DATASET = args.train_dataset
TEST_DATASET = args.test_dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights
BEST_WEIGHTS = args.best_weights

# Check if directory does not exist
create_path('experiments/')
create_path(EXP_DIR)
create_path(CKPT_DIR)
create_path(LOG_DIR)
create_path(os.path.join(LOG_DIR, 'train'))
create_path(os.path.join(LOG_DIR, 'test'))

# Set up logger
filename = os.path.join(LOG_DIR, 'logs.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

# Set up Dataset
train_dataset_1 = Dataset(args, 'train', 1)
train_dataset_2 = Dataset(args, 'train', 2)
train_dataset_3 = Dataset(args, 'train', 3)
train_dataset_4 = Dataset(args, 'train', 4)

test_dataset_1 = Dataset(args, 'test', 1)
test_dataset_2 = Dataset(args, 'test', 2)
test_dataset_3 = Dataset(args, 'test', 3)
test_dataset_4 = Dataset(args, 'test', 4)

train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4])

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True
)

test_dataloader_1 = DataLoader(
    dataset=test_dataset_1,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=False
)

test_dataloader_2 = DataLoader(
    dataset=test_dataset_2,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=False
)

test_dataloader_3 = DataLoader(
    dataset=test_dataset_3,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=False
)

test_dataloader_4 = DataLoader(
    dataset=test_dataset_4,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=False
)

# Get JPEG-XL results
if DO_EVAL:
    if EVAL_ALL:
        jpegxl_bpp_1, jpegxl_avg_bpp_1, jpegxl_avg_time_1 = get_jpegxl_result(test_dataloader_1)
    jpegxl_bpp_2, jpegxl_avg_bpp_2, jpegxl_avg_time_2 = get_jpegxl_result(test_dataloader_2)
    jpegxl_bpp_3, jpegxl_avg_bpp_3, jpegxl_avg_time_3 = get_jpegxl_result(test_dataloader_3)
    jpegxl_bpp_4, jpegxl_avg_bpp_4, jpegxl_avg_time_4 = get_jpegxl_result(test_dataloader_4)

############################
# Encode : yd - ud - vd
#       => yb - ub - vb
#       => yc - uc - vc
############################

color_names = ['Y','U','V']
loc_names = ['d', 'b', 'c']

# Set up networks
networks = setup_networks(color_names, loc_names, logging, HIDDEN_UNIT)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the network to GPU if possible
if torch.cuda.is_available():
    network2cuda(networks, device, color_names, loc_names)

# Set up Loss Functions
lf_L1 = torch.nn.L1Loss(reduction='mean')
lf_ce = torch.nn.CrossEntropyLoss(reduction='mean')

# Load the pretrained model if exists
init_epoch = 0
best_metrics = 1000

if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS))
    init_epoch = checkpoint['epoch_idx']
    LR = checkpoint['lr']

    for color in color_names:
        path_name = 'network_' + color + '_' + WEIGHTS
        checkpoint = torch.load(os.path.join(CKPT_DIR, path_name))
        for loc in loc_names:
            networks[color][loc].load_state_dict(checkpoint['network_' + color + '_' + loc])
    logging.info('Recover completed. Current epoch = #%d' % (init_epoch))

# Create Optimizer / Scheduler
optimizers = setup_optimizers(networks, color_names, loc_names, LR)
schedulers = setup_schedulers(optimizers, color_names, loc_names, DECAY_STEP, DECAY_RATE)

step_num = init_epoch % DECAY_STEP
for _ in range(step_num):
    schedulers_step(schedulers, color_names, loc_names)

# Training/Testing the network
n_batches = len(train_dataloader)

# Constant for masking
constant = {}
for color in color_names:
    if color == 'Y':
        sym_num = 511
    else:
        sym_num = 1021
    constant[color] = var_or_cuda(torch.zeros([1, sym_num, int(CROP_SIZE/2), int(CROP_SIZE/2)]), device=device)
    constant[color][:,0,:,:] = 100000

for epoch_idx in range(init_epoch+1, N_EPOCHS):

    # Metric holders
    losses = get_AverageMeter(color_names, loc_names)
    bitrates = get_AverageMeter(color_names, loc_names)

    total_bitrates = {}
    for color in color_names:
        total_bitrates[color] = AverageMeter()

    # Network to train mode
    network_set(networks, color_names, loc_names, set='train')

    # Train for batches
    for batch_idx, data in enumerate(tqdm(train_dataloader)):
                    
        img_a, img_b, img_c, img_d, ori_img, _, _, qf = data

        temp = torch.eye(4)
        qf = temp[qf-1]

        # Data to cuda
        [img_a, img_b, img_c, img_d, qf] = img2cuda([img_a, img_b, img_c, img_d, qf], device)
        imgs = abcd_unite(img_a, img_b, img_c, img_d, color_names)

        # Inputs / Ref imgs / GTs
        inputs = get_inputs(imgs, color_names, loc_names)
        ref_imgs = get_refs(imgs, color_names)
        gt_imgs = get_gts(imgs, color_names, loc_names)

        for loc in loc_names:
            for color in color_names:
                # Feed to network
                cur_network = networks[color][loc]
                cur_inputs = inputs[color][loc]
                cur_gt_img = gt_imgs[color][loc]
                cur_ref_img = ref_imgs[color]
                cur_optimizer = optimizers[color][loc]

                # Low Frequency Compressor
                pred_L, q_res_L, error_var_map, error_var_th, mask_L, pmf_logit_L = cur_network(cur_inputs, qf, cur_gt_img, cur_ref_img, frequency='low', mode='train')
                mask_H = 1-mask_L

                # High Frequency Compressor Input
                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                # High Frequency Compresor
                pred_H, q_res_H, pmf_logit_H = cur_network(input_H, qf, cur_gt_img, cur_ref_img, frequency='high', mode='train')

                # Prediction Loss
                pred_L_loss = lf_L1(mask_L*cur_gt_img, mask_L*pred_L)
                pred_H_loss = lf_L1(mask_H*cur_gt_img, mask_H*pred_H)

                pred_loss = pred_L_loss + pred_H_loss
                pred_loss *= LAMBDA_PRED

                # Bitrate Loss
                pmf_logit_L = pmf_logit_L * mask_L + constant[color] * mask_H
                pmf_logit_H = pmf_logit_H * mask_H + constant[color] * mask_L
                q_res_L = q_res_L * mask_L
                q_res_H = q_res_H * mask_H

                br_L_loss = lf_ce(pmf_logit_L, q_res_L.squeeze(1))
                br_H_loss = lf_ce(pmf_logit_H, q_res_H.squeeze(1))
                br_loss = br_L_loss + br_H_loss
                br_loss *= LAMBDA_BR

                bits_L = estimate_bits(sym=q_res_L, pmf=torch.nn.Softmax(dim=1)(pmf_logit_L), mask=mask_L)
                bits_H = estimate_bits(sym=q_res_H, pmf=torch.nn.Softmax(dim=1)(pmf_logit_H), mask=mask_H)
                bits = bits_L.item() + bits_H.item()
                bitrate = bits / (4*CROP_SIZE*CROP_SIZE)

                # Error Variance Loss
                ev_loss = lf_L1(error_var_map, torch.abs(cur_gt_img - pred_L))
                ev_loss *= LAMBDA_EV

                # Total Loss
                loss = pred_loss + br_loss + ev_loss

                # Optimize
                cur_optimizer.zero_grad()
                loss.backward()
                cur_optimizer.step()

                loss = loss.item()

                # Update holders
                losses[color][loc].update(loss)
                bitrates[color][loc].update(bitrate)
                total_bitrates[color].update(bitrate)

    # Step scheduler
    schedulers_step(schedulers, color_names, loc_names)

    # Print Epoch Measures
    if epoch_idx % PRINT_EVERY == 0:
        log_color_info(logging, bitrates, color_names, loc_names, epoch_idx, N_EPOCHS)
        log_total_info(logging, total_bitrates, color_names, epoch_idx, optimizers, N_EPOCHS)

    # Evaluate
    if DO_EVAL and (epoch_idx % EVAL_EVERY == 0):

        if EVAL_ALL:
            bitrates_1, enc_times_1 = evaluate(args, logging, networks, test_dataloader_1, device, color_names, loc_names, jpegxl_bpp_1, jpegxl_avg_bpp_1, jpegxl_avg_time_1, epoch_idx)
        bitrates_2, enc_times_2 = evaluate(args, logging, networks, test_dataloader_2, device, color_names, loc_names, jpegxl_bpp_2, jpegxl_avg_bpp_2, jpegxl_avg_time_2, epoch_idx)
        bitrates_3, enc_times_3 = evaluate(args, logging, networks, test_dataloader_3, device, color_names, loc_names, jpegxl_bpp_3, jpegxl_avg_bpp_3, jpegxl_avg_time_3, epoch_idx)
        bitrates_4, enc_times_4 = evaluate(args, logging, networks, test_dataloader_4, device, color_names, loc_names, jpegxl_bpp_4, jpegxl_avg_bpp_4, jpegxl_avg_time_4, epoch_idx)                

        if EVAL_ALL:
            bitrates = bitrates_1['total'].avg() + bitrates_2['total'].avg() + bitrates_3['total'].avg() + bitrates_4['total'].avg()
        else:
            bitrates = bitrates_2['total'].avg() + bitrates_3['total'].avg() + bitrates_4['total'].avg()

        if bitrates <= best_metrics:
            output_path = os.path.join(CKPT_DIR, BEST_WEIGHTS)
            best_metrics = bitrates

            if EVAL_ALL:
                best_metrics_1 = bitrates_1
            best_metrics_2 = bitrates_2
            best_metrics_3 = bitrates_3
            best_metrics_4 = bitrates_4

            torch.save({
                'epoch_idx': epoch_idx,
                'lr' : optimizers['Y']['d'].param_groups[0]["lr"],
                'best_metrics' : best_metrics
            }, output_path)

            for color in color_names:
                path_name = 'network_' + color + '_' + BEST_WEIGHTS
                output_path = os.path.join(CKPT_DIR, path_name)
                torch.save({
                    'network_' + color + '_d': networks[color]['d'].state_dict(),
                    'network_' + color + '_b': networks[color]['b'].state_dict(),
                    'network_' + color + '_c': networks[color]['c'].state_dict()
                }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)

        # Print Best Test Dataset Results
        if EVAL_ALL:
            log_dataset_info(logging, best_metrics_1, jpegxl_avg_bpp_1, enc_times_1, jpegxl_avg_time_1, color_names, loc_names, 'Best')
        log_dataset_info(logging, best_metrics_2, jpegxl_avg_bpp_2, enc_times_2, jpegxl_avg_time_2, color_names, loc_names, 'Best')
        log_dataset_info(logging, best_metrics_3, jpegxl_avg_bpp_3, enc_times_3, jpegxl_avg_time_3, color_names, loc_names, 'Best')
        log_dataset_info(logging, best_metrics_4, jpegxl_avg_bpp_4, enc_times_4, jpegxl_avg_time_4, color_names, loc_names, 'Best')


    # Save Current Model
    if epoch_idx % SAVE_EVERY == 0:

        output_path = os.path.join(CKPT_DIR, WEIGHTS)

        torch.save({
            'epoch_idx': epoch_idx,
            'lr' : optimizers['Y']['d'].param_groups[0]["lr"]
        }, output_path)

        for color in color_names:
            path_name = 'network_' + color + '_' + WEIGHTS
            output_path = os.path.join(CKPT_DIR, path_name)
            torch.save({
                'network_' + color + '_d': networks[color]['d'].state_dict(),
                'network_' + color + '_b': networks[color]['b'].state_dict(),
                'network_' + color + '_c': networks[color]['c'].state_dict()
            }, output_path)

        logging.info('Saved checkpoint to %s ...' % output_path)