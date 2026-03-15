import argparse
import logging
import os
import sys

import joblib



sys.path.append(".")

from utils.dcrf import DenseCRF
from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.model_wsddn import network as network
# from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
# from utils import cam_helper as cam_helper
from utils.pyutils import AverageMeter, format_tabs, format_tabs_multi_metircs, setup_logger
from utils.camutils import cam_to_label, multi_scale_cam2_v2

parser = argparse.ArgumentParser()
# parser.add_argument("--bkg_thre", default=0.5, type=float, help="work_dir")
parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_seg", default=0.12, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_mygo", default=1.0, type=float, help="w_mygo")
# parser.add_argument("--wsddn_topk", default=300, type=int, help="w_mygo")
parser.add_argument('--ot_loss', default=False, action="store_true", help="w_mygo")
parser.add_argument('--ot_n', default=4, type=int, help="w_mygo")
parser.add_argument('--logit_scale', default=4, type=int, help="w_mygo")
parser.add_argument("--cls_sc_t", default=0.0, type=float, help="w_mygo")

parser.add_argument("--model_path", default="model_iter_20000.pth", type=str, help="model_path")
parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--update_prototype", default=600, type=int, help="begin to update prototypes")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
parser.add_argument("--cam_scales", nargs='+', type=float, default=[1.0], help="multi_scales for seg")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")

### knowledge extraction
parser.add_argument('--proto_m', default=0.9, type=float, help='momentum for computing the momving average of prototypes')
parser.add_argument("--temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--base_temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--base_temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--momentum", default=0.999, type=float, help="momentum")
parser.add_argument('--ctc-dim', default=768, type=int, help='embedding dimension')
parser.add_argument('--moco_queue', default=4608, type=int, help='queue size; number of negative samples')
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--save_ckpt", default=False, action="store_true", help="save_ckpt")
parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--weight", default="", type=str, help="model weight")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")
parser.add_argument("--decoder", default='ASSP', type=str, help="dataset folder")
### dataset utils
parser.add_argument("--data_folder", default='./dataset/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=64, type=int, help="crop_size for local view")
parser.add_argument('--ncrops', default=12, type=int, help='number of crops')
parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=2, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--debug', default=False, type=bool,)
parser.add_argument('--lamda', default=100, type=float)
parser.add_argument('--seg_t', default=0.3, type=float)
parser.add_argument('--seg_iter', default=8000, type=int)
parser.add_argument("--infer_set", default="train", type=str, help="infer_set")
parser.add_argument("--wsddn_topk", default=300, type=int, help="w_mygo")
# parser.add_argument("--enable", default=300, type=int, help="w_mygo")

#######消融参数######
parser.add_argument("--no-ema", action="store_false", dest="ema", help="Disable auxiliary augmentation")
parser.add_argument("--no-pro", action="store_false", dest="pro", help="Disable auxiliary augmentation")
parser.add_argument("--no-pre-mask", action="store_false", dest="pre_mask", help="Disable auxiliary augmentation")
parser.add_argument("--no-ctmd", action="store_false", dest="ctmd", help="Disable auxiliary augmentation")

parser.add_argument("--no-mil", action="store_false", dest="mil", help="Disable auxiliary augmentation")
def _validate(model=None, data_loader=None, args=None, branch=1):
    model.eval()
    print(args.cam_scales)


    base_dir = args.model_path.split("checkpoint")[0]
    cam_dir = os.path.join(base_dir, "cam_img", args.infer_set)
    cam_aux_dir = os.path.join(base_dir, "cam_img_aux", args.infer_set)

    os.makedirs(cam_aux_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    color_map = plt.get_cmap("jet")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        preds, gts, cams, cams_aux,cam1 = [], [], [], [],[]

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            img = imutils.denormalize_img(inputs)[0].permute(1, 2, 0).cpu().numpy()

            inputs = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            # _cams = multi_scale_cam2_v2(model, inputs, args.cam_scales,cls_label=None,args=args)
            wsddn_seg,grad_mask, _cams = multi_scale_cam2_v2(model, inputs, args.cam_scales,cls_label=cls_label,args=args)
            # resized_cam1 = F.interpolate(_cam1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            # cam_label1 = cam_to_label(resized_cam1, cls_label, bkg_thre=1e-2, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            
            # print(_cams)
            resized_cam = [F.interpolate(i, size=labels.shape[1:], mode='bilinear', align_corners=False) for i in _cams]
            cam_label = [cam_to_label(i, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMasks=[grad_mask,wsddn_seg],seg_ts = (0,args.seg_t),ret=True) for i in resized_cam]
            
            gts += list(labels.cpu().numpy().astype(np.int16))
            for i,cam_label_i in enumerate(cam_label):
                if idx == 0:
                    cams.append([cam_label_i[1].cpu().numpy().astype(np.int16)])
                else:
                    cams[i] += list(cam_label_i[1].cpu().numpy().astype(np.int16))
            # print(cam_label[1][0])
            cam_np = torch.max(cam_label[1][0][0], dim=0)[0].cpu().numpy()
            cam_rgb = color_map(cam_np)[:, :, :3] * 255
            alpha = 0.6
            cam_rgb = alpha * cam_rgb + (1 - alpha) * img
            imageio.imsave(os.path.join(cam_dir, name[0] + ".jpg"), cam_rgb.astype(np.uint8))
            # np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg":cam_label[1][0].cpu().numpy()})
    cam_score = []
    for cam_i in cams:
        
        cam_score.append(evaluate.scores(gts, cam_i))
# tab_results = format_tabs(cam_score+[seg_score], name_list=[f'CAM{i}'for i in range(len(cam_score))] + [ "Seg_Pred"], cat_list=voc.class_list)
    return format_tabs(cam_score, name_list=[f'CAM{i}'for i in range(len(cam_score))], cat_list=voc.class_list)
def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')
    
    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=1,        # 3
        bi_xy_std=140,  # 121, 140
        bi_rgb_std=1,   # 5, 5
        bi_w=7,         # 4, 5
    )
    print(post_processor)
    def _job(i):

        name = name_list[i]
        # print(name)
        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        if os.path.exists(logit_name):
           os.remove(logit_name)
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)
        # pred = fill_connected_zeros_large_image(pred)
        # imageio.imsave(args.segs_dir + "/" + name + ".png", np.squeeze(pred).astype(np.uint8))
        # imageio.imsave(args.segs_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label
    
    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score


def validate(args=None):

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    model = network(
        args,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=True,
        # pooling=args.pooling,
        aux_layer=-3,
    ).cuda()

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
        

    model.load_state_dict(state_dict=new_state_dict)
    model.eval()

    results = _validate(model=model, data_loader=val_loader, args=args, branch=1)
    torch.cuda.empty_cache()

    print(results.draw())
    crf_proc()
    return True


if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints/")[0]
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')
    args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_cams/logits", args.infer_set)
    os.makedirs(args.logits_dir, exist_ok=True)
    setup_logger(filename=os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_cams/results.log"))

    validate(args=args)
