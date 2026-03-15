import logging
import os

from matplotlib import pyplot as plt
import psutil
import wandb
# from pytorch_grad_cam.grad_cam import GradCAM
from utils.pyutils import AverageMeter, format_tabs
import torch
from tqdm import tqdm
import torch.distributed as dist
from utils import evaluate
from utils.camutils import cam_to_label, multi_scale_cam2_v2
import numpy as np
import torch.nn.functional as F
from datasets import voc
import matplotlib.patches as mpatches
from torchvision.utils import make_grid
def build_validation(model=None, data_loader=None, args=None,n_iter=None,grad_cam =None):

    preds, gts, cams, cams_aux,cam1 = [], [], [], [],[]
    model.eval()
    avg_meter = AverageMeter()
    pred_dir = os.path.join(args.pred_dir, f"{n_iter}P")
    os.makedirs(pred_dir, exist_ok=True)
    hist_segs = np.zeros((args.num_classes, args.num_classes))
    # target_layers = [model.module.encoder.blocks[-1].mlp.fc2]
    # grad_cam = GradCAM(model=model,target_layers=target_layers)
    if args.grad_t =='none':
        grad_cam = None
    with torch.no_grad():
        # process = psutil.Process(os.getpid())
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            origin_input = inputs.squeeze(0)
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            (   cls, segs, fmap, cls_aux,

            backbone_feat,cls_token,cls_token_logit,backbone_weight,outputs_seg_masks, cls_wsddn,_
         ) = model(inputs,args = args)
            
        
            

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})
            wsddn_seg,grad_mask, _cams = multi_scale_cam2_v2(model, inputs, args.cam_scales,cls_label=cls_label,args=args,grad_cam=grad_cam)            
            
            resized_cam = [F.interpolate(i, size=labels.shape[1:], mode='bilinear', align_corners=False) for i in _cams]
            if n_iter > args.seg_iter:
                cam_label = [cam_to_label(i, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMasks=[grad_mask,wsddn_seg],seg_ts = (args.grad_t,args.seg_t)) for i in resized_cam]
            else:
                cam_label = [cam_to_label(i, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,auxiliaryMasks=[grad_mask],seg_ts = (args.grad_t,)) for i in resized_cam]
            del wsddn_seg,grad_mask, _cams
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            seg_pred = torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16)
            
            gt = labels.cpu().numpy().astype(np.int16)
            hist_segs = evaluate.incremental_scores(hist_segs, gt, seg_pred, args.num_classes)
            # if args.debug:
            #     print("seg:",evaluate.scores(gts[-1], preds[-1])['miou'])
            
            for i,cam_label_i in enumerate(cam_label):
                if idx == 0:
                    cams.append(np.zeros((args.num_classes, args.num_classes)))
                    cams[i] = evaluate.incremental_scores(cams[i], gt, cam_label_i.cpu().numpy().astype(np.int16), args.num_classes)
                else:
                    cams[i] = evaluate.incremental_scores(cams[i], gt, cam_label_i.cpu().numpy().astype(np.int16), args.num_classes)
            # if args.debug:
            #     cc = evaluate.scores(gts[-1].squeeze(), cams[1][-1].squeeze())['miou']
            #     print("cam:",cc)
            
            
    
            if  idx %10 == 1 and args.debug:
                seg_mask = F.interpolate(
                    outputs_seg_masks, 
                    size=labels.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                ).sigmoid() > 0.1  # 形状: [batch_size, num_masks, H, W]

                mask_tensor = seg_mask[0].float()  # 形状: [num_masks, H, W]

                image_counter = 1


                start_idx = 0  # 起始索引
                step = 100       # 每个网格的掩膜数量
                if start_idx >= mask_tensor.shape[0]:
                    print(f"Start index {start_idx} 超出掩膜数量范围 {mask_tensor.shape[0]}.")
                else:
                    for i in range(start_idx, mask_tensor.shape[0], step):
                        n_masks = min(step, mask_tensor.shape[0] - i)
                        
                        current_masks = mask_tensor[i:i+n_masks]  # 形状: [n_masks, H, W]
                        
                        current_masks = current_masks.unsqueeze(1)  # [n_masks, 1, H, W]
                        
                        current_masks_rgb = current_masks.repeat(3, 1, 1, 1)  # [n_masks, 3, H, W]
                        center_h, center_w = current_masks.shape[2] // 2, current_masks.shape[3] // 2
                        current_masks_rgb[:, 0, center_h-1:center_h+2, center_w] = 1.0  # 红色垂直线
                        current_masks_rgb[:, 0, center_h, center_w-1:center_w+2] = 1.0  # 红色水平线
                        current_masks_rgb[:, 1:, center_h-1:center_h+2, center_w] = 0.0
                        current_masks_rgb[:, 1:, center_h, center_w-1:center_w+2] = 0.0
                        grid = make_grid(
                            current_masks_rgb, 
                            nrow=10,           # 每行的掩膜数量
                            padding=2,         # 掩膜之间的间距
                            normalize=False,   # 不进行归一化，因为已经是RGB
                            pad_value=1        # 间距颜色（1 为白色）
                        )
                        
                        # 将网格转换为 NumPy 数组以供绘图
                        grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # 形状: [H, W, C]
                        
                        # 使用 matplotlib 绘制网格图像
                        plt.figure(figsize=(10, 10))
                        plt.imshow(grid_np)
                        plt.axis('off')
                        plt.title(f'Image {image_counter}: Masks {i+1} to {i + n_masks}')
                        
                        plt.tight_layout()
                        
                        # 保存图像
                        save_path = os.path.join(pred_dir, f"seg_{i}_{name[0]}.png")
                        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        
                        print(f"已保存网格图像: {save_path}")
                        
                        # 增加图像计数器
                        image_counter += 1
            if idx % 10 == 0 and args.debug:
                pred = cam_label[1].cpu().squeeze(0)
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])

                # 转换 mean 和 std 到合适的形状 (C, 1, 1) 以便进行广播操作
                mean = mean[:, None, None]
                std = std[:, None, None]

                # 执行反归一化
                def unnormalize(image):
                    # image 的形状应该是 (C, H, W)
                    # 检查是否在 CPU 上，如果不是，先移动到 CPU
                    if image.is_cuda:
                        image = image.cpu()
                    
                    # 反归一化
                    image = image * std + mean
                    
                    # 转换为 numpy 数组
                    image = image.numpy()
                    

                    image = image.transpose(1, 2, 0)

                    image = np.clip(image, 0, 1)
                    
                    return image

                origin_input = unnormalize(origin_input)
                CLASS_NAME = {
                    0: "Background",
                    1: "Aeroplane",
                    2: "Bicycle",
                    3: "Bird",
                    4: "Boat",
                    5: "Bottle",
                    6: "Bus",
                    7: "Car",
                    8: "Cat",
                    9: "Chair",
                    10: "Cow",
                    11: "Diningtable",
                    12: "Dog",
                    13: "Horse",
                    14: "Motorbike",
                    15: "Person",
                    16: "Pottedplant",
                    17: "Sheep",
                    18: "Sofa",
                    19: "Train",
                    20: "TVmonitor",
                    255: "Special"  # 添加 255 类别
                }

                # 类别数量
                n_classes = 21

                # 为每个类别分配固定的颜色
                fixed_colors = [
                    (0,(0, 0, 0)),         # Background - 黑色
                    (1,(0.8, 0.2, 0.2)),   # Aeroplane - 红色
                    (2,(0.2, 0.8, 0.2)),   # Bicycle - 绿色
                    (3,(0.2, 0.2, 0.8)),   # Bird - 蓝色
                    (4,(0.8, 0.8, 0.2)),   # Boat - 黄色
                    (5,(0.8, 0.2, 0.8)),   # Bottle - 紫色
                    (6,(0.2, 0.8, 0.8)),   # Bus - 青色
                    (7,(0.5, 0.5, 0.5)),   # Car - 灰色
                    (8,(1.0, 0.5, 0.0)),   # Cat - 橙色
                    (9,(0.5, 0.0, 0.5)),   # Chair - 紫红色
                    (10,(0.0, 0.5, 0.5)),   # Cow - 蓝绿色
                    (11,(0.5, 0.5, 0.0)),   # Diningtable - 橄榄色
                    (12,(0.75, 0.25, 0.25)),# Dog - 红棕色
                    (13,(0.25, 0.75, 0.25)),# Horse - 亮绿色
                    (14,(0.25, 0.25, 0.75)),# Motorbike - 亮蓝色
                    (15,(0.75, 0.75, 0.25)),# Person - 亮黄色
                    (16,(0.75, 0.25, 0.75)),# Pottedplant - 亮紫色
                    (17,(0.25, 0.75, 0.75)),# Sheep - 亮青色
                    (18,(0.1, 0.1, 0.1)),   # Sofa - 深灰色
                    (19,(0.3, 0.3, 0.3)),   # Train - 中灰色
                    (20,(0.6, 0.6, 0.6)),    # TVmonitor - 亮灰色
                    (255,((1.0, 1.0, 1.0))),    # Special - 白色 (255 类别)
                ]
                # colors = [(0, '#FF0000'),  # 红色
                #           (1, '#00FF00'),  # 绿色
                #           (2, '#0000FF')]  # 蓝色
                voc_colors = np.array([
                    [0, 0, 0],        # 0=背景
                    [128, 0, 0],      # 1=飞机
                    [0, 128, 0],      # 2=自行车
                    [128, 128, 0],    # 3=鸟
                    [0, 0, 128],      # 4=船
                    [128, 0, 128],    # 5=瓶子
                    [0, 128, 128],    # 6=公交车
                    [128, 128, 128],  # 7=汽车
                    [64, 0, 0],       # 8=猫
                    [192, 0, 0],      # 9=椅子
                    [64, 128, 0],     # 10=牛
                    [192, 128, 0],    # 11=餐桌
                    [64, 0, 128],     # 12=狗
                    [192, 0, 128],    # 13=马
                    [64, 128, 128],   # 14=摩托车
                    [192, 128, 128],  # 15=人
                    [0, 64, 0],       # 16=盆栽
                    [128, 64, 0],     # 17=羊
                    [0, 192, 0],      # 18=沙发
                    [128, 192, 0],    # 19=火车
                    [0, 64, 128],      # 20=电视
                    [255,255,255]     # 255=?
                ], dtype=np.uint8)
                def visualize_segmentation(segmentation, colors=voc_colors):
                    # segmentation: (H, W) 包含类别索引的分割图
                    height, width = segmentation.shape
                    color_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    for class_id, color in enumerate(colors):
                        if class_id == 21:
                            mask = segmentation == 255
                        else: 
                            mask = segmentation == class_id
                        color_segmentation[mask] = color
                    
                    return color_segmentation
                # 增加一个颜色用于表示255类 (白色)
                # 为每个类别创建一个图例标签
                patches = [mpatches.Patch(color=fixed_colors[i][1], label=CLASS_NAME[key]) for i,key in enumerate(CLASS_NAME)]


                # 使用 imshow 展示分割结果
                plt.figure(figsize=(24, 7))

                # 绘制原始图像
                plt.subplot(1, 5, 1)
                plt.imshow(origin_input)
                plt.title('Original Image')
                plt.axis('off')

                # 绘制真实的分割图
                plt.subplot(1, 5, 2)
                # plt.imshow(labels.squeeze(0).cpu().numpy().astype(np.int16), cmap=cmap)
                colored_segmentation = visualize_segmentation(labels.squeeze(0).cpu())
                plt.imshow(colored_segmentation)
                plt.title('Ground Truth')
                plt.axis('off')
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                # 绘制预测的分割图
                plt.subplot(1, 5, 3)
                colored_segmentation = visualize_segmentation(pred)
                plt.imshow(colored_segmentation)
                # plt.imshow(pred, cmap=cmap)
                plt.title('Cam')
                plt.axis('off')
                
                plt.subplot(1, 5, 4)
                colored_segmentation = visualize_segmentation(torch.argmax(resized_segs, dim=1).squeeze(0).cpu().numpy().astype(np.int16))
                plt.imshow(colored_segmentation)
                plt.title('Seg')
                plt.axis('off')
                
                plt.subplot(1, 5, 5)
                colored_segmentation = visualize_segmentation(cam_label[2].cpu().squeeze(0).cpu().numpy().astype(np.int16))
                plt.imshow(colored_segmentation)
                plt.title('Seg')
                plt.axis('off')
                # 保存和显示图像
                plt.tight_layout()
                plt.savefig(os.path.join(pred_dir, f"{name[0]}"))
                plt.close()

    cls_score = avg_meter.pop('cls_score')
    #_f1_wsddn = avg_meter.pop('wsddn_score')
    seg_score = evaluate.compute_final_scores(hist_segs, args.num_classes)
    # cam_score1 = evaluate.scores(gts, cam1)
    cam_score = []
    for cam_i in cams:
        cam_score.append(evaluate.compute_final_scores(cam_i,args.num_classes))
        # cam_aux_score = evaluate.scores(gts, cams_aux)
    model.train()

    tab_results = format_tabs(cam_score+[seg_score], name_list=[f'CAM{i}'for i in range(len(cam_score))] + [ "Seg_Pred"], cat_list=voc.class_list)
    # if args.local_rank == 0:
    #     logging.info("val cls token loss: %.6f" % (avg_meter.pop('wsddn_score')))
    if args.local_rank == 0:

                all_rows = tab_results._rows

                headers = tab_results._header
                data_rows = all_rows

                for row in data_rows:
                    item_name = row[0]
                    for header, value in zip(headers[1:], row[1:]): 
                        key = f"{item_name}_{header}" 
                        wandb.log({key:float(value)}, step=n_iter)

                logging.info("val cls score: %.6f" % (cls_score))
                #logging.info("val wsddn score: %.6f" % (wsddn_score))
                logging.info("\n"+tab_results.draw())
    return