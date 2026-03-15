import argparse
import os
import sys
import logging
sys.path.append("")
from collections import OrderedDict
import imageio.v2 as imageio
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from datasets import coco
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from model.model_wsddn import network as network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs_multi_metircs, setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--query_decoder_layers', type=int,default=2)
parser.add_argument('--ot_n', default=4, type=int, help="w_mygo")
parser.add_argument('--mask_decoder_layers', type=int,default=1)

parser.add_argument('--debug', default=False, type=bool, help="w_mygo")
parser.add_argument('--ot_loss', default=False, type=bool, help="w_mygo")
parser.add_argument("--decoder", default="ASSP", type=str, help="infer_set")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--model_path", default="model_iter_80000.pth", type=str, help="model_path")

parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--scales", default=[1.0,1.25,1.5], help="multi_scales for seg")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--img_folder", default='/home/data/COCO2014/coco', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/home/data/COCO2014/coco/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--wsddn_topk", default=300, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
# parser.add_argument("--ot_loss", help="random index")

parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument('--ctc-dim', default=768, type=int, help='embedding dimension')
parser.add_argument('--moco_queue', default=4608, type=int, help='queue size; number of negative samples')
parser.add_argument('--mask_source', default='cam_aux', type=str, help='queue size; number of negative samples')
parser.add_argument('--proto_m', default=0.9, type=float, help='momentum for computing the momving average of prototypes')

parser.add_argument("--cls_sc_t", default=0.0, type=float, help="temp")
parser.add_argument("--temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--base_temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--base_temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--momentum", default=0.999, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--master_port", default=29501, type=int, help="master_port")
parser.add_argument("--nproc_per_node", default=8, type=int, help="nproc_per_node")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def _validate(pid, model=None, dataset=None, args=None):

    model.eval()
    data_loader = DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    if args.local_rank == 0:
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >=")
    else:
        pbar = enumerate(data_loader)
    with torch.no_grad():
        model.cuda()

        gts, seg_pred = [], []

        for idx, data in pbar:

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            _, _, h, w = inputs.shape
            seg_list = []
            for sc in args.scales:
                _h, _w = int(h*sc), int(w*sc)

                _inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat)[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                seg = segs[:1,...] + segs[1:,...].flip(-1)

                seg_list.append(seg)
            seg_stack = torch.stack(seg_list, dim=0)
            # seg = seg_stack[0:].mean(dim = 0)
            seg = torch.max(torch.mean(seg_stack[:2],dim=0),seg_stack[0:].max(dim = 0)[0])
            # seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
            seg_save = F.interpolate(seg, size=[int(dim * 0.5) for dim in labels.shape[1:]], mode='bilinear', align_corners=False)
            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg":seg_save.cpu().numpy()})

    # Gather predictions from all processes
    if args.local_rank == 0:
        gather_gts = [None for _ in range(dist.get_world_size())]
        gather_seg_pred = [None for _ in range(dist.get_world_size())]
    else:
        gather_gts = None
        gather_seg_pred = None

    dist.gather_object(gts, gather_gts, dst=0)
    dist.gather_object(seg_pred, gather_seg_pred, dst=0)

    if args.local_rank == 0:
        # Concatenate the lists
        all_gts = []
        all_seg_pred = []
        for gt_list in gather_gts:
            all_gts.extend(gt_list)
        for seg_pred_list in gather_seg_pred:
            all_seg_pred.extend(seg_pred_list)

        seg_score = evaluate.scores(all_gts, all_seg_pred, num_classes=81)
        logging.info('raw_seg_score:')
        metrics_tab = format_tabs_multi_metircs([seg_score], ["confusion","precision","recall",'iou'], cat_list=coco.class_list)
        logging.info("\n"+metrics_tab)
        return seg_score
    else:
        return None

def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    if "val" in args.infer_set:
        images_path = os.path.join(args.img_folder, "val")
        labels_path = os.path.join(args.label_folder, "val")
    elif "train" in args.infer_set:
        images_path = os.path.join(args.img_folder, "train")
        labels_path = os.path.join(args.label_folder, "train")

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=1,        # 3
        bi_xy_std=140,  # 121, 140
        bi_rgb_std=1,   # 5, 5
        bi_w=7,         # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        if len(image.shape)<3:
            image = np.stack((image, image, image), axis=-1)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label
    
    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds,81)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=coco.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score

def validate(args=None):
    if True:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.backend, )
        val_dataset = coco.CocoSegDataset(
            img_dir=args.img_folder,
            label_dir=args.label_folder,
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

        model = network(args,
            backbone=args.backbone,
            num_classes=args.num_classes,
            pretrained=True,
            aux_layer = -3
        )

        trained_state_dict = torch.load(args.model_path, map_location="cpu")
        if 'state_dict' in trained_state_dict:
            trained_state_dict = trained_state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.to(torch.device(args.local_rank))
        model.load_state_dict(state_dict=new_state_dict, strict=True)
        model.eval()
        model = DistributedDataParallel(model, device_ids=[args.local_rank],)
        n_gpus = dist.get_world_size()
        split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]
        
        seg_score = _validate(pid=args.local_rank, model=model, dataset=split_dataset, args=args,)
        torch.cuda.empty_cache()

        # Synchronize all processes
        dist.barrier()

    # Only the main process proceeds to CRF post-processing
    if args.local_rank == 0:
        crf_score = crf_proc()
    else:
        crf_score = None
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints/")[0]
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')
    args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)
    setup_logger(filename=os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/results.log"))

    print(args)
    validate(args=args)
