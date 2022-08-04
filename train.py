# import torch
from torch.utils.data import DataLoader

import logging
import os
from tqdm import tqdm

from config import parse_args
from utils.average_meter import AverageMeter
from utils.data_loaders import *
from utils.log_helpers import *
from utils.helpers import *
from utils.data_transformer import *
from eval import evaluate

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
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
N_EPOCHS = args.epochs
LR = args.lr
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

DO_EVAL = args.do_eval
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

# Dataloader
if 'SIDD' in TRAIN_DATASET:
    train_dataset = SIDD_Dataset(args, 'train')
elif 'mit' in TRAIN_DATASET:
    train_dataset = MIT_Dataset(args, 'train')

if 'SIDD' in TEST_DATASET:
    test_dataset = SIDD_Dataset(args, 'test')
elif 'mit' in TEST_DATASET:
    test_dataset = MIT_Dataset(args, 'test')

Collate_ = DivideCollate()

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True,
    collate_fn=Collate_
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=NUM_WORKERS,
    shuffle=False,
    collate_fn=Collate_
)

# Get FLIF results
if DO_EVAL:
    flif_bpp_msb, flif_bpp_lsb, flif_avg_bpp_msb, flif_avg_bpp_lsb, flif_avg_time = get_flif_result(test_dataloader)

############################
# Encode : yd - ud - vd
#       => yb - ub - vb
#       => yc - uc - vc
############################

color_names = ['Y','U','V','D']
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
best_metrics = {}
best_metrics['total'] = {}
best_metrics['total']['MSB'] = AverageMeter()
best_metrics['total']['LSB'] = AverageMeter()
best_metrics['total']['MSB'].update(1000)
best_metrics['total']['LSB'].update(1000)

if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS))
    init_epoch = checkpoint['epoch_idx']
    best_metrics = checkpoint['best_metrics']
    LR = checkpoint['lr']

    for color in color_names:
        path_name = 'network_' + color + '_' + WEIGHTS
        checkpoint = torch.load(os.path.join(CKPT_DIR, path_name))
        for loc in loc_names:
            networks[color][loc]['MSB'].load_state_dict(checkpoint['network_MSB' + color + '_' + loc])
            networks[color][loc]['LSB'].load_state_dict(checkpoint['network_LSB' + color + '_' + loc])
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
        total_bitrates[color] = {}
        total_bitrates[color]['MSB'] = AverageMeter()
        total_bitrates[color]['LSB'] = AverageMeter()

    # Network to train mode
    network_set(networks, color_names, loc_names, set='train')

    # Train for batches
    for batch_idx, data in enumerate(tqdm(train_dataloader)):
                    
        img_a_msb, img_b_msb, img_c_msb, img_d_msb, img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb, ori_img, _, _ = data

        # Data to cuda
        [img_a_msb, img_b_msb, img_c_msb, img_d_msb] = img2cuda([img_a_msb, img_b_msb, img_c_msb, img_d_msb], device)
        [img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb] = img2cuda([img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb], device)
        imgs_msb = abcd_unite(img_a_msb, img_b_msb, img_c_msb, img_d_msb, color_names)
        imgs_lsb = abcd_unite(img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb, color_names)

        # Inputs / Ref imgs / GTs
        inputs_msb = get_inputs(imgs_msb, color_names, loc_names)
        ref_imgs_msb = get_refs(imgs_msb, color_names)
        gt_imgs_msb = get_gts(imgs_msb, color_names, loc_names)

        inputs_lsb = get_inputs(imgs_lsb, color_names, loc_names)
        ref_imgs_lsb = get_refs(imgs_lsb, color_names)
        gt_imgs_lsb = get_gts(imgs_lsb, color_names, loc_names)

        for loc in loc_names:
            for color in color_names:
                # ==================================== MSB ====================================#
                # Feed to network
                cur_network = networks[color][loc]['MSB']
                cur_inputs = inputs_msb[color][loc]
                cur_gt_img = gt_imgs_msb[color][loc]
                cur_ref_img = ref_imgs_msb[color]
                cur_optimizer = optimizers[color][loc]['MSB']

                # Low Frequency Compressor
                pred_L, q_res_L, error_var_map, error_var_th, mask_L, pmf_logit_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='train')
                mask_H = 1-mask_L

                # High Frequency Compressor Input
                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                # High Frequency Compresor
                pred_H, q_res_H, pmf_logit_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='train')

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
                losses[color][loc]['MSB'].update(loss)
                bitrates[color][loc]['MSB'].update(bitrate)
                total_bitrates[color]['MSB'].update(bitrate)


                # ==================================== LSB ====================================#
                # Feed to network
                cur_network = networks[color][loc]['LSB']
                cur_inputs = inputs_lsb[color][loc]
                cur_gt_img = gt_imgs_lsb[color][loc]
                cur_ref_img = ref_imgs_lsb[color]
                cur_optimizer = optimizers[color][loc]['LSB']

                # Low Frequency Compressor
                pred_L, q_res_L, error_var_map, error_var_th, mask_L, pmf_logit_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='train')
                mask_H = 1-mask_L

                # High Frequency Compressor Input
                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                # High Frequency Compresor
                pred_H, q_res_H, pmf_logit_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='train')

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
                losses[color][loc]['LSB'].update(loss)
                bitrates[color][loc]['LSB'].update(bitrate)
                total_bitrates[color]['LSB'].update(bitrate)
    # Step scheduler
    schedulers_step(schedulers, color_names, loc_names)

    # Print Epoch Measures
    if epoch_idx % PRINT_EVERY == 0:
        log_color_info(logging, bitrates, color_names, loc_names, epoch_idx, N_EPOCHS)
        log_total_info(logging, total_bitrates, color_names, epoch_idx, optimizers, N_EPOCHS)

    # Evalaute
    if DO_EVAL and (epoch_idx % EVAL_EVERY == 0):

        if epoch_idx % (1*EVAL_EVERY) == 0:
            PRINT_INDIVIDUAL = True
        else:
            PRINT_INDIVIDUAL = False

        bitrates, enc_times = evaluate(args, logging, networks, test_dataloader, device, color_names, loc_names, flif_bpp_msb, flif_bpp_lsb, flif_avg_bpp_msb, flif_avg_bpp_lsb, flif_avg_time, PRINT_INDIVIDUAL)

        if bitrates['total']['MSB'].avg() + bitrates['total']['LSB'].avg() <= best_metrics['total']['MSB'].avg() + best_metrics['total']['LSB'].avg():
            output_path = os.path.join(CKPT_DIR, WEIGHTS)
            best_metrics = bitrates

            torch.save({
                'epoch_idx': epoch_idx,
                'lr' : optimizers['Y']['d']['MSB'].param_groups[0]["lr"],
                'best_metrics' : best_metrics
            }, output_path)

            for color in color_names:
                path_name = 'network_' + color + '_' + WEIGHTS
                output_path = os.path.join(CKPT_DIR, path_name)
                torch.save({
                    'network_MSB' + color + '_d': networks[color]['d']['MSB'].state_dict(),
                    'network_MSB' + color + '_b': networks[color]['b']['MSB'].state_dict(),
                    'network_MSB' + color + '_c': networks[color]['c']['MSB'].state_dict(),
                    'network_LSB' + color + '_d': networks[color]['d']['LSB'].state_dict(),
                    'network_LSB' + color + '_b': networks[color]['b']['LSB'].state_dict(),
                    'network_LSB' + color + '_c': networks[color]['c']['LSB'].state_dict()                  
                }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)

        # Print Best Test Dataset Results
        log_dataset_info(logging, best_metrics, flif_avg_bpp_msb, flif_avg_bpp_lsb, enc_times, flif_avg_time, color_names, loc_names, 'Best')

    if not DO_EVAL and (epoch_idx % SAVE_EVERY == 0):

        output_path = os.path.join(CKPT_DIR, WEIGHTS)

        torch.save({
            'epoch_idx': epoch_idx,
            'lr' : optimizers['Y']['d']['MSB'].param_groups[0]["lr"],
            'best_metrics' : best_metrics
        }, output_path)

        for color in color_names:
            path_name = 'network_' + color + '_' + WEIGHTS
            output_path = os.path.join(CKPT_DIR, path_name)
            torch.save({
                'network_MSB' + color + '_d': networks[color]['d']['MSB'].state_dict(),
                'network_MSB' + color + '_b': networks[color]['b']['MSB'].state_dict(),
                'network_MSB' + color + '_c': networks[color]['c']['MSB'].state_dict(),
                'network_LSB' + color + '_d': networks[color]['d']['LSB'].state_dict(),
                'network_LSB' + color + '_b': networks[color]['b']['LSB'].state_dict(),
                'network_LSB' + color + '_c': networks[color]['c']['LSB'].state_dict() 
            }, output_path)

        logging.info('Saved checkpoint to %s ...' % output_path)

