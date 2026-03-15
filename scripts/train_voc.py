import argparse
import datetime
import logging
import os
import random
import sys
from matplotlib import pyplot as plt
import wandb
import sys
sys.path.append('.')
# from pytorch_grad_cam import GradCAM

sys.path.append("")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.losses import (ClsTokenLoss, MyGo_Loss, MyGo_Loss2, MyGo_Loss3,  get_masked_ptc_loss, get_seg_loss,get_energy_loss, DenseEnergyLoss)
from torch.nn.parallel import DistributedDataParallel
from model.PAR import PAR
from utils import imutils,evaluate
from utils.camutils import (cam_to_label, get_token_mask, label_to_aff_mask, multi_scale_cam2_v2, 
                            refine_cams_with_bkg_v2,)
from utils.vis import get_train_miou
from utils.wsddnutils import get_atten_mask
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from engine import build_network_wsddn, build_optimizer, build_validation
from torch.cuda.amp import autocast, GradScaler
import scipy.ndimage as ndi
parser = argparse.ArgumentParser()
torch.hub.set_dir("./pretrained")
### loss weight
parser.add_argument("--w_ptc", default=0.3, type=float, help="weight for ptc loss")
parser.add_argument("--w_seg", default=0.12, type=float, help="weight for segmentation loss")
parser.add_argument("--w_reg", default=0.05, type=float, help="weight for regression loss")
parser.add_argument("--wsddn", default=2.0, type=float, help="weight for wsddn loss")

parser.add_argument("--bin_t", default=0.7, type=float, help="binary threshold")
parser.add_argument("--w_cpe", default=0.02, type=float, help="weight for cpe loss")
parser.add_argument("--w_seg2", default=2.0, type=float, help="weight for auxiliary segmentation loss")
parser.add_argument("--wsddn_topk", default=300, type=int, help="top k proposals for wsddn")
parser.add_argument('--ot_loss', default=False, action="store_true", help="whether to use optimal transport loss")
parser.add_argument('--seg_aux', default=False, action="store_true", help="use auxiliary segmentation branch")
parser.add_argument('--reg_aux', default=False, action="store_true", help="use auxiliary regression branch")

parser.add_argument('--ot_n', default=4, type=int, help="number of iterations or components for OT")
parser.add_argument("--cls_sc_t", default=0.0, type=float, help="classification score threshold")
parser.add_argument("--seg_warmup", default=2000, type=int, help="warmup iterations for segmentation")
parser.add_argument("--ptc_eps", default=1, type=float, help="epsilon value for ptc")




### training utils
parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--update_prototype", default=2000, type=int, help="begin to update prototypes")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
# parser.add_argument('--no-mil', default=True, action="store_false", help="w_cpe")
### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--pro_high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--pro_low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
# parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.5), help="multi_scales for cam")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--decoder", default='ASSP', type=str, help="pooling choice for patch tokens")


### knowledge extraction
parser.add_argument('--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes')
parser.add_argument("--momentum", default=0.999, type=float, help="momentum")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")
def parse_scales(s):
    try:
        return tuple(map(float, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Scales must be a comma-separated list of numbers.")
def parse_grad_t(value):
    try:
        return float(value)
    except ValueError:
        if value in ['x', 'max','none']:
            return value
        else:
            raise argparse.ArgumentTypeError(f"Invalid value for --grad_t: {value}. Must be 'x', 'max', or a float.")
### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--save_ckpt", default=False, action="store_true", help="save_ckpt")
parser.add_argument("--color_aug", default=True, action="store_true", help="save_ckpt")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--weight", default="", type=str, help="model weight")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")
parser.add_argument("--wandb_name", default='test', type=str, help="dataset folder")

### dataset utils
parser.add_argument("--data_folder", default='./dataset/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=64, type=int, help="crop_size for local view")
parser.add_argument("--cnt_t", default=5, type=int, help="crop_size for local view")
parser.add_argument('--ncrops', default=12, type=int, help='number of crops')
parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=2, type=int, help="samples_per_gpu")
parser.add_argument("--scales", type=parse_scales, default=(0.5, 2), help="random rescale in training")
parser.add_argument("--no-aux_aug", action="store_false", dest="aux_aug", help="Disable auxiliary augmentation")
### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=3e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=2, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--debug',  default=False,action="store_true")
parser.add_argument('--lamda', default=100, type=float)
parser.add_argument('--seg_t', default=0.3, type=float)
parser.add_argument('--grad_t', default="none", type=parse_grad_t)

parser.add_argument('--seg_iter', default=8000, type=int)
parser.add_argument("--cam_scales", 
                        nargs='+', 
                        type=float, 
                        default=[1.0, 0.5, 0.75, 1.5], 
                        help="multi_scales for cam (provide as space-separated values)")

parser.add_argument('--prog_high_t', default=0.5, type=float)
parser.add_argument('--prog_low_t', default=0.3, type=float)

#######ablation args######
parser.add_argument("--no-ema", action="store_false", dest="ema", help="Disable auxiliary augmentation")
parser.add_argument("--no-pro", action="store_false", dest="pro", help="Disable auxiliary augmentation")
parser.add_argument("--no-pre-mask", action="store_false", dest="pre_mask", help="Disable auxiliary augmentation")
parser.add_argument("--no-mask_exp", action="store_false", dest="mask_exp", help="Disable auxiliary augmentation")

parser.add_argument("--no-ctmd", action="store_false", dest="ctmd", help="Disable auxiliary augmentation")
parser.add_argument("--mygo_sup", action="store_true", dest="mygo_sup", help="Disable auxiliary augmentation")
parser.add_argument("--no-mask-stop", action="store_false", dest="mask_stop", help="Disable auxiliary augmentation")
parser.add_argument("--pro_fuse", action="store_true", dest="pro_fuse", help="Disable auxiliary augmentation")
parser.add_argument("--wsddn_loss", type=str,default="sum", help="Disable auxiliary augmentation")
parser.add_argument("--no-mil", action="store_false", dest="mil", help="Disable auxiliary augmentation")


# parser.add_argument("--clstoken_loss", action="store_true", dest="clstoken_loss", help="Disable auxiliary augmentation")
parser.add_argument("--clstoken_loss", default='cos', type=str)




def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args=None):
    DEBUG = args.debug
    if args.local_rank == 0:
        wandb.init(
                project="wsss",
                name=args.wandb_name,
                config=vars(args)  # 把argparse的超参数传入config
            )
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    time1 = None
    device = torch.device(args.local_rank)
    if args.ot_loss:
        get_seg_crme_loss = MyGo_Loss3(topk=args.wsddn_topk)
    else:
        get_seg_crme_loss = MyGo_Loss2(cnt_t = args.cnt_t,iou_score_t= args.bin_t,topk=args.wsddn_topk,cls_sc_t=args.cls_sc_t,high_t=args.prog_high_t,low_t=args.prog_low_t)
    ### build model 
    model, param_groups = build_network_wsddn(args,get_seg_crme_loss)
    # print(model)
    model.to(device)
    if args.weight != "":
        # original saved file with DataParallel
        state_dict = torch.load(args.weight)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        # load params
        model.load_state_dict(new_state_dict,strict=True)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    ### build dataloader 

    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
        color_aug = args.color_aug,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
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
    
    get_cls_token_loss = ClsTokenLoss(model.module.encoder.embed_dim,args.num_classes - 1).to(device)
    ### build optimizer 
    for param in list(get_cls_token_loss.parameters()):
        param_groups[2].append(param)
    optim = build_optimizer(args,param_groups)
    scaler = GradScaler()
    logging.info('\nOptimizer: \n%s' % optim)

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()
    miou1 = miou2 =miou3= 0

    if args.grad_t =='none':
        grad_cam = None
    else:
        target_layers = [model.module.encoder.blocks[-1].mlp.fc2]
        grad_cam = None
        
        # grad_cam = GradCAM(model=model,target_layers=target_layers)
    for n_iter in range(args.max_iters):
        if args.debug:

            n_iter = n_iter + 6000
        try:
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image,seg_mask = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image,seg_mask = next(train_loader_iter)
        
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        # cams, cams_aux, auxiliaryMask = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales, cls_label=cls_label,args=args)
        if n_iter < args.seg_iter:
            
            auxiliaryMask,grad_mask ,(cams_aux, cams,prototypes_cam) = multi_scale_cam2_v2(model, inputs=inputs, scales=args.cam_scales, args=args,grad_cam=grad_cam,cls_label=cls_label)
            # GradCAM
            cams_aux, cams,prototypes_cam = cams_aux.detach(), cams.detach(),prototypes_cam.detach()
            #cams *= prototypes_cam
            if args.pro_fuse:
                cams = torch.sum(torch.stack([0.5 * cams,0.5 * prototypes_cam]), dim = 0)
            valid_cam, asd = cam_to_label(cams, cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMasks=[grad_mask],seg_ts = (args.grad_t,))
            valid_cam_aux , pseudo_label = cam_to_label(cams_aux, cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        else:
            #if not gumbel_sofmax:
            #    cams_aux, cams,prototypes_cam = cams_aux.detach(), cams.detach(),prototypes_cam.detach()
            auxiliaryMask ,grad_mask,(cams_aux, cams,prototypes_cam) = multi_scale_cam2_v2(model, inputs=inputs, scales=args.cam_scales,cls_label=cls_label, args=args,mask_circle = False,grad_cam=grad_cam)
            cams_aux, cams,prototypes_cam = cams_aux.detach(), cams.detach(),prototypes_cam.detach()
            if args.pro_fuse:
                cams = torch.sum(torch.stack([0.5 * cams,0.5 * prototypes_cam]), dim = 0)
            #cams *= prototypes_cam
            valid_cam, asd = cam_to_label(cams, cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMasks=[grad_mask,auxiliaryMask],seg_ts = (args.grad_t,args.seg_t))
            valid_cam_aux , pseudo_label = cam_to_label(cams_aux, cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMasks=[auxiliaryMask] if args.aux_aug else None,seg_ts = (args.seg_t,))

            
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )

        refined_pseudo_label_aux = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam_aux, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        # refined_pseudo_label
        ### aff loss from ToCo, https://github.com/rulixiang/ToCo
        resized_cams_aux = F.interpolate(cams_aux, size=(28,28), mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        # if n_iter == 19:
        #     print(20)
        cls_token_mask = get_token_mask(pseudo_label_aux,args.num_classes,cls_label.detach())
        if n_iter >= -1:
            atten_mask = None
        else:
            atten_mask = get_atten_mask(fmap.detach())


        (   cls, segs, fmap, cls_aux,
            backbone_feat,cls_token,cls_token_logit,backbone_weight,outputs_seg_masks, cls_wsddn,prototypes
        ) = model(
            inputs,
            label_idx=label_idx,
            n_iter=n_iter,
            atten_mask=atten_mask,
            cls_label=cls_label,
            aug_img = None,
            args = args
        )
        ### cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        ## wsddn loss
        if not args.mil:
            cls_loss_wsddn = F.multilabel_soft_margin_loss(cls_wsddn.sum(dim = 1), cls_label)
        elif args.wsddn_loss == "sum":
            cls_loss_wsddn = F.binary_cross_entropy(cls_wsddn.sum(dim = 1), cls_label.detach())
        elif args.wsddn_loss == "max":
            cls_loss_wsddn = F.binary_cross_entropy(cls_wsddn.max(dim = 1)[0], cls_label.detach())
        if args.clstoken_loss == "cos":
            cpe_loss = get_masked_ptc_loss(fmap.detach(), cls_token_mask.flatten(-2),cls_token)
        elif args.clstoken_loss == "linear":
            cpe_loss = get_cls_token_loss(cls_token,cls_label)
        elif args.clstoken_loss == "sum":
            cpe_loss = get_cls_token_loss(cls_token,cls_label,sum=True)
        # cls_token_loss = get_cls_token_loss(cls_token,cls_label)

        # ot 
        # ot_mask = get_cls_token_loss.get_seg_mask(cls_tokens=cls_token,cls_label=cls_label,pseudo_label=None,fmap=fmap,refined_pseudo_label=refined_pseudo_label)
        # refine_ot_mask = refine_cams_with_bkg_v2(par, inputs_denorm, cams=ot_mask, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)


        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
        if args.pro: # I channged here, for 72.3 need change to <
            valid_pro, asd = cam_to_label(prototypes_cam, cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
            refined_pro_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_pro, cls_labels=cls_label,  high_thre=args.pro_high_thre, low_thre=args.pro_low_thre, ignore_index=args.ignore_index, img_box=img_box, )

            crme_loss = get_seg_crme_loss(outputs_seg_masks, cls_wsddn.detach(), cls_label, refined_pro_label.detach(), topk=args.wsddn_topk,mask_circle = ((n_iter <= args.seg_iter and args.pre_mask) or args.mask_stop),mask_exp = args.mask_exp,fin_size=refined_pseudo_label.shape[-2:],mask_circle_r = 1 + (1 - np.cos((n_iter - args.seg_warmup) / (args.seg_iter - args.seg_warmup) * np.pi)) / 2 * 4)
            get_seg_crme_loss.update_expand_time(1 + (1 - np.cos((n_iter - args.seg_warmup) / (args.seg_iter - args.seg_warmup) * np.pi)) / 2 * 9)
        else:
            crme_loss = get_seg_crme_loss(outputs_seg_masks, cls_wsddn.detach(), cls_label.detach(), refined_pseudo_label.detach(), topk=args.wsddn_topk,mask_circle = True if n_iter <= args.seg_iter else False,fin_size=refined_pseudo_label.shape[-2:])
        if args.mygo_sup:
            seg_loss = get_seg_loss(segs, refined_pro_label.detach().type(torch.long), ignore_index=args.ignore_index)
        else:
            seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        w_cpe = args.w_cpe
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)
        if args.seg_aux:
            seg_loss_aux = get_seg_loss(segs, refined_pseudo_label_aux.type(torch.long), ignore_index=args.ignore_index)
            seg_loss = (seg_loss + seg_loss_aux) / 2
        if args.reg_aux:
            reg_aux_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label_aux, img_box=img_box, loss_layer=loss_layer)
            reg_loss = (reg_loss + reg_aux_loss) / 2
        if n_iter <= args.seg_warmup:
            loss =   1.0 * cls_loss + 1.0 * cls_loss_aux  + args.w_ptc * ptc_loss + args.wsddn * cls_loss_wsddn + w_cpe * cpe_loss
        elif n_iter <= args.seg_iter:
            loss =   1.0 * cls_loss + 1.0 * cls_loss_aux  + w_cpe * cpe_loss + args.w_ptc * ptc_loss + args.w_seg  * seg_loss + args.w_seg2 * crme_loss +args.wsddn *  cls_loss_wsddn + args.w_reg * reg_loss
        else:
            loss =   1.0 * cls_loss + 1.0 * cls_loss_aux  + w_cpe * cpe_loss + args.w_ptc * ptc_loss + args.w_seg  * seg_loss + args.w_seg2 * crme_loss + args.wsddn * cls_loss_wsddn + args.w_reg * reg_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cpe_loss': cpe_loss.item(),
            'crme_loss': crme_loss.item() if isinstance(crme_loss,torch.Tensor) else crme_loss,
            'cls_score': cls_score.item(),
            'cls_loss_wsddn':cls_loss_wsddn.item(),
        })

        optim.zero_grad()

        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:
            
            delta, eta = cal_eta(time0,time1 if time1 is not None else time0, n_iter + 1, args.max_iters,args.log_iters)
            
            time1 = datetime.datetime.now().replace(microsecond=0)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                log_data = {
                    "Iter": n_iter + 1,
                    "Elapsed": delta,
                    "ETA": eta,
                    "LR": f"{cur_lr:.3e}",
                    "cls_loss": avg_meter.pop('cls_loss'),
                    "cls_loss_aux": avg_meter.pop('cls_loss_aux'),
                    "ptc_loss": avg_meter.pop('ptc_loss'),
                    "seg_loss": avg_meter.pop('seg_loss'),
                    "cpe_loss": avg_meter.pop('cpe_loss'),
                    "crme_loss": avg_meter.pop('crme_loss'),
                    "cls_loss_wsddn": avg_meter.pop('cls_loss_wsddn'),
                }


                logging.info(
                    f"Iter: {log_data['Iter']}; Elapsed: {log_data['Elapsed']}; ETA: {log_data['ETA']}; LR: {log_data['LR']}; "
                    f"cls_loss: {log_data['cls_loss']:.4f}, "
                    f"cls_loss_aux: {log_data['cls_loss_aux']:.4f}, "
                    f"ptc_loss: {log_data['ptc_loss']:.4f}, "
                    f"seg_loss: {log_data['seg_loss']:.4f}, "
                    f"cpe_loss: {log_data['cpe_loss']:.4f}, "
                    f"crme_loss: {log_data['crme_loss']:.4f}, "
                    f"cls_loss_wsddn: {log_data['cls_loss_wsddn']:.4f}"
                )
                wandb.log(log_data,step=n_iter)


        if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1:
            
            logging.info(f"miou1:{miou1/args.eval_iters}  miou2:{miou2/args.eval_iters} miou3:{miou3/args.eval_iters}")
            miou1 = miou2 = miou3 = 0
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
                build_validation(model=model, data_loader=val_loader, args=args,n_iter=n_iter,grad_cam=grad_cam)

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
