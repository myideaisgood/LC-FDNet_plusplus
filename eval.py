import torch

from tqdm import tqdm

from config import parse_args
from torch.utils.data import DataLoader
from utils.average_meter import AverageMeter
from utils.data_loaders import *
from utils.log_helpers import *
from utils.helpers import *

def get_qf(img):
    [_, H, W, _] = list(img.shape)

    imgsize = H * W

    DEFAULT = 2040*1356

    if imgsize >= DEFAULT:
        return torch.tensor([1], dtype=torch.int64)
    elif imgsize >= DEFAULT/4:
        return torch.tensor([2], dtype=torch.int64)
    elif imgsize >= DEFAULT/16:
        return torch.tensor([3], dtype=torch.int64)    
    else:
        return torch.tensor([4], dtype=torch.int64)


def evaluate(args, logging, networks, test_dataloader, device, color_names, loc_names, jpegxl_bpp, jpegxl_avg_bpp, jpegxl_avg_time, epoch_idx):

    torch.cuda.empty_cache()

    with torch.no_grad():
        network_set(networks, color_names, loc_names, set='eval')

        bitrates = get_AverageMeter(color_names, loc_names)
        bitrates['total'] = AverageMeter()

        enc_times = {}
        for loc in loc_names:
            enc_times[loc] = AverageMeter()

        try:
            for batch_idx, data in enumerate(tqdm(test_dataloader)):

                img_a, img_b, img_c, img_d, ori_img, img_name, padding, _ = data

                qf = get_qf(ori_img)

                temp = torch.eye(4)
                qf = temp[qf-1]

                # Data to cuda
                [img_a, img_b, img_c, img_d, qf] = img2cuda([img_a, img_b, img_c, img_d, qf], device)
                imgs = abcd_unite(img_a, img_b, img_c, img_d, color_names)

                [_, H, W, _] = list(ori_img.shape)

                # Inputs / Ref imgs / GTs
                inputs = get_inputs(imgs, color_names, loc_names)
                ref_imgs = get_refs(imgs, color_names)
                gt_imgs = get_gts(imgs, color_names, loc_names)

                for loc in loc_names:

                    start_time = time()

                    for color in color_names:
                        # Feed to network
                        cur_network = networks[color][loc]
                        cur_inputs = inputs[color][loc]
                        cur_gt_img = gt_imgs[color][loc]
                        cur_ref_img = ref_imgs[color]

                        # Low Frequency Compressor
                        _, q_res_L, _, _, mask_L, pmf_softmax_L = cur_network(cur_inputs, qf, cur_gt_img, cur_ref_img, frequency='low', mode='eval')
                        mask_H = 1-mask_L

                        bits_L = estimate_bits(sym=q_res_L, pmf=pmf_softmax_L, mask=mask_L)

                        del q_res_L, pmf_softmax_L
                        torch.cuda.empty_cache()

                        # High Frequency Compressor Input
                        gt_L = mask_L * cur_gt_img
                        input_H = torch.cat([cur_inputs, gt_L], dim=1)

                        # High Frequency Compresor
                        _, q_res_H, pmf_softmax_H = cur_network(input_H, qf, cur_gt_img, cur_ref_img, frequency='high', mode='eval')
                        bits_H = estimate_bits(sym=q_res_H, pmf=pmf_softmax_H, mask=mask_H)

                        bits = bits_L.item() + bits_H.item()
                        bitrate = bits / (H*W)

                        # Update holders
                        bitrates[color][loc].update(bitrate)

                        del q_res_H, pmf_softmax_H
                        torch.cuda.empty_cache()
                    
                    enc_time = time() - start_time
                    enc_times[loc].update(enc_time)
                
                update_total(bitrates, color_names, loc_names)
                
                # Print Test Img Results
                if epoch_idx % 500 == 0:
                    log_img_info(logging, img_name, bitrates, jpegxl_bpp[batch_idx], color_names, loc_names)
            
            # Print Test Dataset Results
            log_dataset_info(logging, bitrates, jpegxl_avg_bpp, enc_times, jpegxl_avg_time, color_names, loc_names)
        
        except Exception as ex:
            logging.info(ex)

    return bitrates, enc_times


if __name__ == '__main__':

    ######### Configuration #########
    ######### Configuration #########
    ######### Configuration #########
    args = parse_args()

    # Design Parameters
    HIDDEN_UNIT = args.hidden_unit

    # Session Parameters
    GPU_NUM = args.gpu_num
    EMPTY_CACHE = args.empty_cache

    # Directory Parameters
    DATASET = args.test_dataset
    TEST_DOWNSCALE_RATIO = args.test_downscale_ratio
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
    LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
    BEST_WEIGHTS = args.best_weights

    # Set up logger
    filename = os.path.join(LOG_DIR, 'logs_eval.txt')
    logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))
        logging.info('\t%15s:\t%s' % (key, value))

    ######### Configuration #########
    ######### Configuration #########
    ######### Configuration #########

    # Set up Dataset
    # Dataloader
    test_dataset = Dataset(args, 'test', TEST_DOWNSCALE_RATIO)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )

    # JPEG-XL result of img a
    jpegxl_bpp, jpegxl_avg_bpp, jpegxl_avg_time = get_jpegxl_result(test_dataloader)

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

    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, BEST_WEIGHTS))

    for color in color_names:
        path_name = 'network_' + color + '_' + BEST_WEIGHTS
        checkpoint = torch.load(os.path.join(CKPT_DIR, path_name))
        for loc in loc_names:
            networks[color][loc].load_state_dict(checkpoint['network_' + color + '_' + loc])
    logging.info('Recover completed.')

    bitrates, enc_times = evaluate(args, logging, networks, test_dataloader, device, color_names, loc_names, jpegxl_bpp, jpegxl_avg_bpp, jpegxl_avg_time, epoch_idx=500)