import argparse
import collections
import datetime
import logging
import os
import random
import sys
import time


sys.path.append("")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import coco
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.losses import (ClsTokenLoss, MyGo_Loss, MyGo_Loss2, MyGo_Loss3,  get_masked_ptc_loss, get_masked_ptc_loss_v2, get_mil_loss, get_seg_loss, get_energy_loss, 
                        LIG_Loss, LIL_Loss, DenseEnergyLoss, get_seg_mask_loss)
from torch.nn.parallel import DistributedDataParallel
from model.PAR import PAR
from utils import imutils,evaluate
from utils.camutils import (cam_to_label, get_token_mask, multi_scale_cam2, label_to_aff_mask, multi_scale_cam2_v2, 
                            refine_cams_with_bkg_v2, assign_csc_tags, cam_to_roi_mask)
from utils.vis import get_train_miou
from utils.wsddnutils import get_atten_mask
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from engine import build_network_wsddn, build_optimizer, build_validation
from torch.cuda.amp import autocast, GradScaler
import scipy.ndimage as ndi
parser = argparse.ArgumentParser()
torch.hub.set_dir("./pretrained")

### loss weight
parser.add_argument("--w_ptc", default=0.2, type=float, help="weight for ptc loss")
parser.add_argument("--w_seg", default=0.12, type=float, help="weight for segmentation loss")
parser.add_argument("--w_reg", default=0.05, type=float, help="weight for regression loss")
parser.add_argument("--w_cpe", default=0.5, type=float, help="weight for cpe loss")

parser.add_argument("--wsddn_topk", default=300, type=int, help="top k proposals for wsddn")
parser.add_argument('--ot_loss', default=False, action="store_true", help="enable optimal transport loss")
parser.add_argument('--ot_n', default=4, type=int, help="number of iterations for Sinkhorn/OT")
parser.add_argument('--logit_scale', default=4, type=int, help="scaling factor for logits")

### training utils
parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=5000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--update_prototype", default=600, type=int, help="begin to update prototypes")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
parser.add_argument("--cam_scales", 
                        nargs='+', 
                        type=float, 
                        default=[1.0, 0.5, 0.75, 1.5], 
                        help="multi_scales for cam (provide as space-separated values)")
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

### dataset utils
parser.add_argument("--img_folder", default='/Data/MSCOCO/JPEGImages', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/home/data/COCO2014/coco/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=64, type=int, help="crop_size for local view")
parser.add_argument('--ncrops', default=12, type=int, help='number of crops')
parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")
parser.add_argument("--spg", default=2, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=7e-7, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--debug', default=False, type=bool,)
parser.add_argument('--lamda', default=100, type=float)
parser.add_argument('--seg_t', default=0.3, type=float)
parser.add_argument('--seg_iter', default=32000, type=int)
parser.add_argument('--resume', type=str)
def smart_resume(ckpt, optimizer, model):
    # Resume training from a partially trained checkpoint
    # best_fitness = 0.0
    n_iter = ckpt['iter'] + 1
    if ckpt['optimizer'] is not None:
        optimizer_stat = ckpt['optimizer']  # optimizer
        # best_fitness = ckpt['best_fitness']
    state_dict = ckpt['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict,strict=True)
    optimizer.load_state_dict(optimizer_stat)
    return n_iter,model,optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args=None):
    DEBUG = args.debug
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    device = torch.device(args.local_rank)

    ### build model 
    model, param_groups = build_network_wsddn(args)
    # print(model)
    model.to(device)
    if args.weight != "":
        # original saved file with DataParallel
        state_dict = torch.load(args.weight)['state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict,strict=True)
    

    ### build dataloader 
    train_dataset = coco.CocoClsDataset(
        img_dir="/home/data/COCO2014/coco",
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = coco.CocoSegDataset(
        img_dir="/home/data/COCO2014/coco",
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)
    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    get_mygo_loss = MyGo_Loss()
    loss_mse = torch.nn.MSELoss()
    if args.ot_loss:
        get_seg_mygo_loss = MyGo_Loss3(topk=args.wsddn_topk)
    else:
        get_seg_mygo_loss = MyGo_Loss2(topk=args.wsddn_topk)
    get_cls_token_loss = ClsTokenLoss().to(device)
    ### build optimizer 
    for param in list(get_cls_token_loss.parameters()):
        param_groups[2].append(param)
    optim = build_optimizer(args,param_groups)
    n_iter_resume = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        n_iter_resume,model,optim = smart_resume(ckpt,optim,model)
        optim.resume(n_iter_resume)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    # scaler = GradScaler()
    logging.info('\nOptimizer: \n%s' % optim)

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()
    miou1 = miou2 =miou3= 0
    
    for n_iter in range(n_iter_resume,args.max_iters):
        # n_iter = n_iter + 50000
        try:
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())

        cls_label = cls_label.to(device, non_blocking=True)

        # cams, cams_aux, auxiliaryMask = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales, cls_label=cls_label,args=args)
        if n_iter < args.seg_iter:

            auxiliaryMask ,(cams_aux, cams,prototypes_cam) = multi_scale_cam2_v2(model, inputs=inputs, scales=args.cam_scales,cls_label=cls_label, args=args)

            valid_cam, asd = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            _ , pseudo_label = cam_to_label(F.interpolate(cams.detach(), size=(28,28), mode="bilinear", align_corners=False), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        else:
            # s = time.time()
            auxiliaryMask ,(cams_aux, cams,prototypes_cam) = multi_scale_cam2_v2(model, inputs=inputs, scales=args.cam_scales,cls_label=cls_label, args=args)
            # e = time.time()
            # print(e-s)
            valid_cam, asd = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMask=auxiliaryMask,seg_t = args.seg_t)
            _ , pseudo_label = cam_to_label(F.interpolate(cams.detach(), size=(28,28), mode="bilinear", align_corners=False), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        # if DEBUG or True:
        #     tt = get_train_miou(asd,seg_mask)
        #     if DEBUG:
        #         print(tt)
        #     miou1 += tt
        ### aff loss from ToCo, https://github.com/rulixiang/ToCo
        resized_cams_aux = F.interpolate(cams_aux, size=(28,28), mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        # cls, segs, fmap, cls_aux, out_q, queue_feats_all, queue_flags_all, prototype = model(inputs,label_idx=label_idx,crops=roi_crops,cls_flags_local=cls_flags_local, n_iter=n_iter)
        if n_iter >= -1:
            atten_mask = None
        else:
            atten_mask = get_atten_mask(fmap.detach())
        # atten_mask = None
        # if n_iter==55:
        #     print("233")
        # s = time.time()
        (   cls, segs, fmap, cls_aux,
            backbone_feat,cls_token,cls_token_logit,backbone_weight,outputs_seg_masks, cls_wsddn,prototypes
        ) = model(
            inputs,
            label_idx=label_idx,
            n_iter=n_iter,
            atten_mask=atten_mask,
            cls_label=cls_label,
            aug_img = None
        )
        # e = time.time()
        # print(e-s)
        ### cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # 使用模型1的特征图监督模型2的特征图
        # loss_feat = loss_mse(backbone_feat.detach(), backbone_feat)
        ## wsddn loss
        cls_loss_wsddn = F.binary_cross_entropy(cls_wsddn.sum(dim = 1), cls_label.float())
        
        cls_token_mask = get_token_mask(pseudo_label_aux,args.num_classes,cls_label)
        cpe_loss = get_masked_ptc_loss(fmap.detach(), cls_token_mask.flatten(-2),cls_token)



        ### seg_loss & reg_loss
    
        
        # cls_token_loss = get_cls_token_loss(cls_token_logit,cls_label)
       
            

        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # crme_loss = get_seg_mygo_loss(outputs_seg_masks, cls_wsddn.detach(), cls_label.detach(), refined_pseudo_label.detach(), topk=args.wsddn_topk)

        w_cpe = args.w_cpe

        if n_iter < 60000 or True: # I channged here, for 72.3 need change to <
            valid_pro, asd = cam_to_label(prototypes_cam.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            refined_pro_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_pro, cls_labels=cls_label,  high_thre=0.9, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
            crme_loss = get_seg_mygo_loss(outputs_seg_masks, cls_wsddn.detach(), cls_label.detach(), refined_pro_label.detach(), topk=args.wsddn_topk,mask_circle = True if n_iter <= args.seg_iter else False,fin_size=refined_pseudo_label.shape[-2:])
        else:
            crme_loss = get_seg_mygo_loss(outputs_seg_masks, cls_wsddn.detach(), cls_label.detach(), refined_pseudo_label.detach(), topk=args.wsddn_topk,mask_circle = True if n_iter <= args.seg_iter else False,fin_size=refined_pseudo_label.shape[-2:])
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)
        if n_iter <= 8000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux  + args.w_cpe * cpe_loss + cls_loss_wsddn 
        elif n_iter <= 12000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux  + args.w_cpe * cpe_loss + cls_loss_wsddn
        elif n_iter <= 16000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux  + args.w_cpe * cpe_loss + cls_loss_wsddn + args.w_ptc * ptc_loss + 1.0 * crme_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux  + args.w_cpe * cpe_loss + cls_loss_wsddn + args.w_ptc * ptc_loss + 1.0 * crme_loss + args.w_seg * seg_loss + args.w_reg * reg_loss
        # if n_iter <= 8000:
        #     loss =   1.0 * cls_loss + 1.0 * cls_loss_aux  + w_cpe * cls_token_loss + args.w_ptc * ptc_loss + cls_loss_wsddn
        # elif n_iter <= args.seg_iter:
        #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux  + w_cpe * cls_token_loss + args.w_seg  * seg_loss+ args.w_ptc * ptc_loss + 2.0 * crme_loss + cls_loss_wsddn
        # else:
        #     # crme_loss = get_seg_mygo_loss(outputs_seg_masks, cls_wsddn.detach(), cls_label.detach(), refined_pseudo_label.detach(), topk=args.wsddn_topk,shape=refined_pseudo_label.shape[1:])
        #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux  + w_cpe * cls_token_loss + args.w_seg  * seg_loss+ args.w_ptc * ptc_loss + 2.0 * crme_loss + cls_loss_wsddn

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cpe_loss': cpe_loss.item(),
            'cls_score': cls_score.item(),
            'cls_loss_wsddn':cls_loss_wsddn.item(),
            'crme_loss': crme_loss.item() if isinstance(crme_loss,torch.Tensor) else crme_loss,

        })
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optim.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optim)
        # scaler.update()
        loss.backward()
        optim.step()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"Parameter {name} was not used in the computation.")
        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info(
                f"Iter: {n_iter + 1}; Elapsed: {delta}; ETA: {eta}; LR: {cur_lr:.3e}; "
                f"cls_loss: {avg_meter.pop('cls_loss'):.4f}, "
                f"cls_loss_aux: {avg_meter.pop('cls_loss_aux'):.4f}, "
                f"ptc_loss: {avg_meter.pop('ptc_loss'):.4f}, "
                f"seg_loss: {avg_meter.pop('seg_loss'):.4f}, "
                f"cpe_loss: {avg_meter.pop('cpe_loss'):.4f}, "
                f"crme_loss: {avg_meter.pop('crme_loss'):.4f}, "
                f"cls_loss_wsddn: {avg_meter.pop('cls_loss_wsddn'):.4f}, "

                )

        if args.save_ckpt and (n_iter + 1) % 5000 == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            logging.info(f'saving {ckpt_name}')
            if args.local_rank == 0:
                torch.save({'iter': n_iter, 'state_dict': model.state_dict(),
					'optimizer': optim.state_dict()}, ckpt_name)
        if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1 and ((n_iter + 1) >= 60000 or args.debug):
            logging.info(f"miou1:{miou1/args.eval_iters}  miou2:{miou2/args.eval_iters} miou3:{miou3/args.eval_iters}")
            miou1 = miou2 = miou3 = 0
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
            val_cls_score, tab_results = build_validation(model=model, data_loader=val_loader, args=args,n_iter=n_iter)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                #logging.info("val wsddn score: %.6f" % (wsddn_score))
                logging.info("\n"+tab_results)

    return True


if __name__ == "__main__":


    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    exp_tag = f'{args.log_tag}_{timestamp_2}'
    args.work_dir = os.path.join(args.work_dir, timestamp_1, exp_tag)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
