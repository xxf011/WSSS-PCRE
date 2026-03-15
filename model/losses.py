import pdb
import time
from typing import List
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
import scipy.ndimage as ndi
from utils.wsddnutils import draw_mask, visualize_tensor_list
from scipy.ndimage import gaussian_filter
from bilateralfilter import bilateralfilter, bilateralfilter_batch
def visualize_feature_map(feature_map):
    """
    Visualize a 4x448x448 feature map as three separate heatmaps.
    
    Args:
    feature_map (torch.Tensor): A tensor of shape (4, 448, 448).
    """
    # if feature_map.shape != (4, 448, 448):
    #     raise ValueError("Input feature map must have shape (4, 448, 448)")

    # Convert the tensor to numpy array for plotting
    feature_map_np = feature_map.detach().cpu().numpy()
    n = feature_map_np.shape[0]
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, n, figsize=(18, 6))
    
    for i in range(n):
        ax = axs[i]
        # Plot the heatmap
        im = ax.imshow(feature_map_np[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'Feature Map {i+1}')
        fig.colorbar(im, ax=ax)

    # Show the plots
    plt.savefig("hotmap.png")
    plt.close()

def visualize_mask(mask):
    """
    可视化给定大小的掩码图像。

    参数:
    size (int): 掩码图像的大小 (size x size)。
    mask_start (tuple): 掩码区域的起始坐标 (y, x)。
    mask_end (tuple): 掩码区域的结束坐标 (y, x)。
    """

    # 显示mask
    if isinstance(mask,torch.Tensor):
        mask = mask.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.title(f'Mask')
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.savefig("mask.png")
    plt.close()

def Sinkhorn( K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T
def ot(image_features,text_features,N,lamda = 1e-2,sim = None):
    if sim is None:
        M,b,_ = image_features.shape
        _,n_cls,_ = text_features.shape
        image_features =  F.normalize(image_features, dim=2) 
        text_features = F.normalize(text_features, dim=2)


        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(M,N,b*n_cls)
        sim = sim.permute(2,0,1)
    else:
        n_cls = 1
        b ,M,N = sim.shape
    wdist = 1.0 - sim
    xx=torch.zeros(b*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
    yy=torch.zeros(b*n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

    with torch.no_grad():
        KK = torch.exp(-wdist / lamda)
        T = Sinkhorn(KK,xx,yy)
    if torch.isnan(T).any():
        return None

    return T

class LIG_Loss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def cal_pos_logit(self,flags,prototype_flags,logits):
        
        mask_all_pos = flags == prototype_flags.T
        cls_index_ = torch.unique(flags)
        cls_index = cls_index_[cls_index_!=-1]
        cls_index = cls_index[cls_index!=0]
        prototype_cls_mask = torch.zeros_like(mask_all_pos).to(flags.device)
        for idx in cls_index:
            col = int(idx - 1)
            prototype_cls_mask[:,col] = 1

        logits_filtered_pos = logits * mask_all_pos
        # filter wrong labled pos pairs
        logits_filtered_all = logits * prototype_cls_mask
        # prevent nan 

        return logits_filtered_pos, logits_filtered_all, mask_all_pos


    def forward(self, output_q, prototypes,flags):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        output_q        :(20,DIM) 10*b,
        prototypes      :(20,DIM) 20
        flags           :(20,1)
        """
        num_cls = prototypes.shape[0]
        b = output_q.shape[0]
        prototypes_flag = torch.arange(1,num_cls+1).reshape(-1,1).to(flags.device) # 1,2,3,...,20 for VOC
        logits = torch.matmul(output_q, prototypes.T)
        logits = torch.div(logits , self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_all = logits - logits_max.detach()
        logits_all = logits

        logits_pos, logits_all, mask_pos_pos = self.cal_pos_logit(flags,prototypes_flag,logits_all)

        # compute log_prob
        exp_logits = torch.exp(logits_all)
        exp_logits = exp_logits * (exp_logits != 1.0000) # only calculate the appeared class
        log_prob = logits_pos - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        logits_pos_mask = logits_pos != 0
        log_prob = log_prob * logits_pos_mask

        loss = torch.tensor([0.0]).to(output_q.device)
        for idx in range(logits_pos.shape[0]):
            exp_logits_value = exp_logits.sum(1, keepdim=True)
            if exp_logits_value[idx] > 0:
                loss += log_prob[idx].sum()
                # loss_num += 1
            else:
                pass

        # compute mean of log-likelihood over positive
        if mask_pos_pos.sum() > 0:
            mean_log_prob_pos = loss / mask_pos_pos.sum()
        else:
            mean_log_prob_pos =  loss / logits_pos.shape[0]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss


class LIL_Loss(nn.Module): 

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def cal_pos_logit(self,flags,queue_flags_all,logits,n_iter):

        mask_all_pos = flags == queue_flags_all.T
        
        idx_m1,_ = np.where(flags.cpu().numpy()==-1)
        # mask_all_pos[idx_0] = 0 # get rid of bkg
        mask_all_pos[idx_m1] = 0 # get rid of cooccurence
    
        # get rid of -1 cls 
        logits[idx_m1] = 0
        # logits[idx_0] = 0

        logits_filtered_pos = logits * mask_all_pos

        # filter wrong labled pos pairs
        if n_iter >= 0: # for stable demtermine 
            for i in range(len(flags)):
                mean_sim = logits_filtered_pos[i].mean()
                _pos_logits_index = torch.where(logits_filtered_pos[i] >= mean_sim)[0]
                _wrong_pos_logits_index = torch.where(logits_filtered_pos[i] < mean_sim)[0]
                if len(_wrong_pos_logits_index) / (len(_pos_logits_index) + 1e-6) > 10:
                    flags[i] = 0 # get rid of noise label 
                    logits_filtered_pos[i] = 0.0

        queue_index  = torch.where(queue_flags_all == -1)[0]
        logits[:,queue_index] = 0.0
        return logits_filtered_pos, logits, flags

    def forward(self, output_q, queue_all, flags, queue_flags_all,n_iter):
        b = output_q.shape[0]
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(output_q, queue_all[b:].T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_all = anchor_dot_contrast - logits_max.detach()
        # mask-out bkg and cooccurence
        logits_pos,logits_all,flags_revised = self.cal_pos_logit(flags,queue_flags_all[b:],logits_all,n_iter)
        # compute log_prob
        exp_logits = torch.exp(logits_all)
        exp_logits = exp_logits * (exp_logits != 1.0000) # only calculate the appeared class
        # only class pair is calculated
        log_prob = logits_pos - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        loss = torch.tensor([0.0]).to(output_q.device)
        for idx in range(logits_pos.shape[0]):
            num_nozero = torch.nonzero(logits_pos).size(0)
            exp_logits_value = exp_logits.sum(1, keepdim=True)
            if num_nozero > 0 and  exp_logits_value[idx] > 0 and log_prob[idx].sum() < 0:
                loss += log_prob[idx].sum() / num_nozero
            else:
                pass

        loss /= logits_pos.shape[0]

        # loss
        loss = - (self.temperature / self.base_temperature) * loss

        return loss,flags_revised


def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_masked_ptc_loss_v2(inputs, mask):
    inputs_cos = inputs.clone()
    inputs_cos[inputs_cos == float('-inf')] = 0
    inputs_cos =F.normalize(inputs_cos, p=2, dim=1, eps=1e-8)
    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss
def euclidean_distance_matrix(batch_data):
    # batch_data 是形状为 (b, d, n) 的张量
    b, d, n = batch_data.shape
    
    # 转置数据以便计算欧氏距离
    batch_data = batch_data.permute(0, 2, 1)  # 变为 (b, n, d)
    
    # 扩展 batch_data 的维度以便进行广播
    batch_data = batch_data.unsqueeze(2)  # 变为 (b, n, 1, d)
    batch_data_t = batch_data.transpose(1, 2)  # 变为 (b, 1, n, d)
    
    # 计算欧氏距离
    distance_matrix = torch.sqrt(torch.sum((batch_data - batch_data_t) ** 2, dim=3))  # 结果为 (b, n, n)
    return distance_matrix

def gaussian_similarity_matrix(distance_matrix, d):
    # 使用高斯核函数将距离矩阵转换为相似度矩阵
    similarity_matrix = torch.exp(-(distance_matrix ** 2) / (2 * d ** 2))
    return similarity_matrix
def inputs_min_max_normalized(inputs):
    inputs_min, _ = torch.min(inputs, dim=2, keepdim=True)
    inputs_max, _ = torch.max(inputs, dim=2, keepdim=True)
    inputs_min_max_normalized = (inputs - inputs_min) / (inputs_max - inputs_min + 1e-8)
    euclidean_distance_matrix = torch.cdist(inputs_min_max_normalized.permute(0, 2, 1), inputs_min_max_normalized.permute(0, 2, 1))
    return euclidean_distance_matrix
class ConvBlockToLogits(nn.Module):
    def __init__(self):
        super(ConvBlockToLogits, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 14, 14)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 7, 7)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 3, 3)
            
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=0),  # Output: (4, 1, 1)
            # nn.BatchNorm2d(4),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)
def gaussian_kernel_matrix(X, Y, sigma=1.0):
    XX = torch.sum(X**2, dim=1, keepdim=True)
    YY = torch.sum(Y**2, dim=1, keepdim=True)
    distances = XX + YY.T - 2 * torch.mm(X, Y.T)
    kernel_matrix = torch.exp(-distances / (2 * sigma**2))
    return kernel_matrix
class ClsTokenLoss(nn.Module):
    def __init__(self,
                 input_dim,num_classes,label_weight: List[float] = [1.0, 0.9, 0.8]
                 ):
        super().__init__()
        self.register_buffer('label_weight', torch.tensor(label_weight, dtype=torch.float32))
        self.N = 3
        # weights = torch.tensor([1.0 / 2.7, 0.9 / 2.7,0.8 / 2.7, 0])  # 权重向量
        self.cls_proj = nn.Linear(input_dim,1)
        # self.conv = ConvBlockToLogits()
        
    def forward(self, cls_tokens, labels,sum = False):
        if sum:
            cls_tokens_logits = cls_tokens.sum(-1)
        else:
            cls_tokens_logits = self.cls_proj(cls_tokens).squeeze(-1)
        loss = F.multilabel_soft_margin_loss(cls_tokens_logits, labels, reduction='mean')

        return loss
    def get_seg_mask(self,cls_tokens,cls_label,pseudo_label,fmap,refined_pseudo_label,lamda = 1e-2,weight = None):
        N = self.N
        B,M,C= cls_tokens.shape
        K = M // N
        H,W = 28,28
        HW = fmap.shape[-1] * fmap.shape[-2]
        cls_tokens = cls_tokens.reshape(B,N,K,C)
        ot_tokens = []
        t_list_total = []
        # weights = self.weights
        # fmap
        # backbone_weight = backbone_weight[:,:,:M,M:].reshape(B,12,N,K,-1)
        loss = torch.tensor(0.0).to(cls_tokens.device)
        tt = 0
        t_list = []
        for b in range(B):
            
            
            

            cls_token_i = cls_tokens[b,]
            feature = fmap[b]
            # feature =  F.interpolate(fmap[b].unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False)
            # sim_head = backbone_weight[b,:,:,cls_id].flatten(0,1).reshape(36,28,28)
            # neg_mask = (batch_iou(sim_head >0,pseudo_label == cls_id +1) ==0).squeeze()
            # true_mask = (batch_iou(sim_head >0,pseudo_label == cls_id +1) > 0.6).squeeze()
            # true_sim = sim_head[true_mask].flatten(-2).T.unsqueeze(0)
            # neg_sim = sim_head[neg_mask].flatten(-2).T.unsqueeze(0)
            # sim = torch.cat([true_sim,neg_sim],dim=-1)
            # N = 1
            T = ot(feature.flatten(-2).permute(2,0,1),cls_token_i.reshape(60,-1).unsqueeze(1),60,lamda=1) 
            if T is None:
                continue
            T = (T * H*W * (60)) - 1
            T = T.reshape(H,W,60).permute(2,0,1).reshape(3,20,H,W)
            # visualize_feature_map(T)
            # T_weight = self.conv(F.interpolate(T.unsqueeze(0), size=(28,28), mode='bilinear', align_corners=False))
            
            T = F.interpolate(T, size=(448, 448), mode='bilinear', align_corners=False)
            T = (1.0 * T[0] + 0.9 * T[1] + 0.8 * T[2]) / 2.7
            # loss += F.binary_cross_entropy_with_logits(T_resized.squeeze(),(refined_pseudo_label[b] == cls_id+1).to(T_resized.dtype))
            # tt += 1
            # T_resized = T_resized.squeeze()

            # T_resized[T_resized < 0.5] = 0
            T = T.relu()
                
                # 归一化到0-1之间
            # T_normalized = (refined_pseudo_label[b] == cls_id + 1)*T_resized
            T_normalized = (T - T.min(dim = 0,keepdim=True)[0]) / (T.max(dim = 0,keepdim=True)[0] - T.min(dim = 0,keepdim=True)[0] + 1e-6)
            # mask_T = (batch_iou(T_normalized > 0.5,refined_pseudo_label[b].unsqueeze(0) == cls_id + 1) > 0.5)
            # if not mask_T.any():
            #     t_list.append(None)
            #     continue
            
            # T_normalized = (refined_pseudo_label[b] == cls_id + 1)*T_normalized
            # T_normalized[T_normalized < 0.9] = 0
            # masked_feature_map = torch.from_numpy(ndi.binary_fill_holes(T_normalized.cpu() > 0.5)).to(T_normalized.device)
            # masked_feature_map = (masked_feature_map & ~(T_normalized > 0.5))
            # T_normalized[masked_feature_map] = 0.71
            
            # in_side_mask = T_normalized > 0.7
            # tt = ot(feature[:,:,in_side_mask].permute(2,0,1),ot_token.unsqueeze(1),N+1,lamda=lamda)
            # tt = tt * in_side_mask.shape[-1]* (N + 1) - 1
            # tt = tt.permute(2,0,1)
            # tt = tt[mask_T].mean(0).squeeze().relu()
            # tt =  tt / tt.max()
            # T_normalized[in_side_mask] = tt


            # masked_feature_map = torch.from_numpy(ndi.binary_fill_holes(T_normalized.cpu() > 0.7)).to(T_normalized.device)
            # masked_feature_map = (masked_feature_map & ~(T_normalized > 0.7))
            # T_normalized[masked_feature_map] = 0.71
            if False:
                # 如果张量在 GPU 上，移动到 CPU 并转换为 NumPy 数组
                tensor= T_normalized 
                if tensor.is_cuda:
                    tensor = tensor.cpu()

                tensor = tensor.detach().numpy()  # 转换为 NumPy 数组

                # 设置网格布局参数
                cols = 5  # 每行显示5个图
                rows = 4  # 总共4行，共20个类别

                # 创建子图
                fig, axes = plt.subplots(rows, cols, figsize=(25, 20))  # 根据需要调整大小

                # 遍历每个类别并绘制
                for i in range(20):
                    row = i // cols
                    col = i % cols
                    ax = axes[row, col]
                    
                    prob_map = tensor[i]  # 获取第 i 个类别的概率图
                    
                    # 显示概率图
                    img = ax.imshow(prob_map, cmap='viridis')  # 你可以更换其他 colormap，例如 'hot', 'jet'
                    
                    # 添加标题（如果有类别名称，可以替换 'Class {i+1}'）
                    ax.set_title(f'class {i+1}', fontsize=14)
                    
                    # 去除坐标轴
                    ax.axis('off')
                    
                    # 添加颜色条（可选）
                    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


                plt.tight_layout()


                plt.savefig('probability_maps.png', dpi=300)
                plt.close()
            t_list.append(T_normalized)
            # 类间ot
            # cls_ids = cls_label[b].nonzero()[:,0]
            # if cls_ids.shape[0] == 1 or True:
            #     t_list_total.extend(t_list)
            #     continue
            # ot_token = cls_tokens[b,:,cls_ids,:].sum(0)
            # in_class_mask = (refined_pseudo_label[b] >0) & (refined_pseudo_label[b] <21)
            # CLS_NUM = ot_token.shape[0]
            # T = ot(feature[:,:,in_class_mask].permute(2,0,1),ot_token.unsqueeze(1),CLS_NUM,lamda=lamda)
            # T = ((T * in_class_mask.sum() * CLS_NUM) - 1).relu()
            # T = (T - T.min(dim = -1,keepdim=True)[0]) / (T.max(dim = -1,keepdim=True)[0] - T.min(dim = -1,keepdim=True)[0])
            # T[T < 0.7] = 0.0
            # for list_id,cls_id in enumerate(cls_ids):
            #     if t_list[list_id] is None:
            #         t_list_total.append(None)
            #         continue
            #     new_t = t_list[list_id]
            #     new_t[in_class_mask] = torch.max(T[0,:,list_id],new_t[in_class_mask])
            #     new_t = (new_t- new_t.min(dim = -1,keepdim=True)[0]) / (new_t.max(dim = -1,keepdim=True)[0] - new_t.min(dim = -1,keepdim=True)[0])
            #     t_list_total.append(new_t)
            #     # print("??")
            
                
            # # t_list_total.append(*t_list)
            
                
        return torch.stack(t_list)
                





def get_masked_ptc_loss(inputs, mask,cls_token = None):
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)
    def cos_sim(x,y = None):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        if y is None:
            cos_sim = torch.matmul(x.transpose(1,2), x)
        else:
            y = F.normalize(y, p=2, dim=-1, eps=1e-8)
            cos_sim = torch.matmul(y, x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs,cls_token)
    # if True:
    # a = inputs_cos.reshape(inputs_cos.shape[0],-1,28,28)
    # draw_mask(a.cpu(),"affmask")


    pos_mask = mask == 1
    neg_mask = mask == 0

    # pos_mask = torch.from_numpy(gaussian_filter(pos_mask.float().cpu().reshape(b,h*w,h,w), sigma=0.3).reshape(b, h*w, h*w)).to(inputs.device)
    # neg_mask = torch.from_numpy(gaussian_filter(neg_mask.float().cpu().reshape(b,h*w,h,w), sigma=0.3).reshape(b, h*w, h*w)).to(inputs.device)
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss

def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (0.5 * bg_loss + 0.5 * fg_loss)
# def get_seg_loss_bce(pred, label, ignore_index=255):
#     ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
#     bg_label = label.clone()
#     bg_label[label!=0] = ignore_index
#     bg_sum = (bg_label != ignore_index).long().sum()
#     # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
#     bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
#     fg_label = label.clone()
#     fg_label[label==0] = ignore_index
#     fg_sum = (fg_label != ignore_index).long().sum()
#     # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
#     fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

#     return (bg_loss + fg_loss) * 0.5
def get_seg_bce_loss(pred, target, ignore_index=255):
    """
    计算二分类的分割损失，使用二元交叉熵损失（BCELoss）。
    
    参数：
    - pred: 已经过 sigmoid 激活的预测值 (batch_size, 1, height, width)
    - label: 标签 (batch_size, height, width)，前景为1，背景为0
    - ignore_index: 被忽略的标签索引值，默认为255
    
    返回：
    - loss: 平均二分类交叉熵损失
    """
    pos_mask = (target == 1)  # 正样本掩码
    neg_mask = (target == 0)  # 负样本掩码

    # 分别计算正负样本的 BCE 损失
    pos_loss = F.binary_cross_entropy(pred, target.float(), reduction='none') * (pos_mask)
    neg_loss = F.binary_cross_entropy(pred, target.float(), reduction='none') * (neg_mask)

    # 计算正负样本的平均损失
    pos_loss_mean = pos_loss.sum() / (pos_mask.sum()+ 1e-6)
    neg_loss_mean = neg_loss.sum() / (neg_mask.sum()+ 1e-6)

    # 输出正样本和负样本的平均损失
    # print("Positive Loss:", pos_loss_mean.item())
    # print("Negative Loss:", neg_loss_mean.item())

    # 如果需要组合损失，可以按权重组合
    total_loss = 0.5 * pos_loss_mean + 0.5 * neg_loss_mean  # 或者自定义加权，例如 0.7 * pos_loss_mean + 0.3 * neg_loss_mean
    # print("Total Loss:", total_loss.item())
    
    return total_loss
def get_seg_mask_loss(pred, cross_weight, size=None):
    b,q,h,w = pred.shape
    label = cross_weight >= cross_weight.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    pred = pred.view(b, q, h*w)
    # pred = pred.permute(0, 2, 1).reshape(2, 784, 784)

    # 将 label 转换为浮点类型
    label = label.float()

    # 使用 BCEWithLogitsLoss 计算损失
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(pred, label)
    return loss
    
    
def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)

    if img_box is not None:
        crop_mask = torch.zeros_like(pred_prob[:, 0, ...])
        for idx, coord in enumerate(img_box):
            crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    else:
        crop_mask = torch.ones_like(pred_prob[:, 0, ...])

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()
def get_mil_loss(scores,label):
    scores_sum = torch.sum(scores, dim = 1)
    loss = F.binary_cross_entropy(scores_sum, label,reduction="mean")
    return loss
    
class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None

    
class MyGo_Loss(nn.Module): 

    def __init__(self, h = 28, w = 28):
        super().__init__()
        self.H = h
        self.W = w
        self.criterion_logits = nn.BCELoss()

    def forward(self,score_map, score, label,gt):
        # step 1 归一化score_map
        _, B, N, hw = score_map.shape
        assert self.H * self.W == hw
        resize_map = score_map.mean(0).reshape(B, N, self.H, self.W).relu()
        max_values = torch.max(resize_map, dim=-1, keepdim=True)[0]
        max_values = torch.max(max_values, dim=-2, keepdim=True)[0]

        # 使用最大值进行归一化
        resize_map = resize_map / (max_values + 1e-6)
        cam_list = []
        cls_label = label.nonzero()
        loss = 0
        for b,cls_id in cls_label:
            cls_sc = score[b,:,cls_id]
            sorted_original_indices = torch.where(cls_sc > 1e-2)[0][cls_sc[cls_sc > 1e-2].argsort(descending=True)]
            fin_list = []
            if sorted_original_indices.shape[0] < 1:
                sorted_original_indices = torch.tensor([cls_sc.argmax()])
                fin_list = [sorted_original_indices]
            else:
                max_index = sorted_original_indices[0]
                ori_mask = resize_map[b,max_index].unsqueeze(0) > 0.7
                queue_mask = resize_map[b,sorted_original_indices] > 0.7
                iou_score = (batch_iou(ori_mask,queue_mask) + batch_dice(ori_mask,queue_mask))/2
                fin_list.append(sorted_original_indices[(iou_score > 0.5).squeeze(0)])
                fin_list = [torch.cat(fin_list)]
                while True:
                    ori_mask = resize_map[b,fin_list[0]] > 0.7
                    queue_mask = resize_map[b,sorted_original_indices] > 0.7
                    iou_score = (batch_iou(ori_mask,queue_mask) + batch_dice(ori_mask,queue_mask))/2
                    idx = sorted_original_indices[(iou_score>0.5).nonzero(as_tuple=True)[1].unique()]
                    if fin_list[0].shape[0] == idx.shape[0]:
                        break
                    else:
                        fin_list.append(idx)
                        fin_list = [torch.cat(fin_list).unique()]
            pred = F.interpolate(resize_map[b,fin_list[0]].unsqueeze(1), size=gt.shape[1:], mode='bilinear', align_corners=False).mean(0)
            gt_mask = (gt[b] == cls_id +1)

            loss += self.criterion_logits(pred.squeeze(0), gt_mask.float())
        loss /= cls_label.shape[0]
            

            
        # step 2 分batch处理w
        return loss
    def expand_mask(self, mask, iou_t = 0.7):
        pass
class MyGo_Loss2(nn.Module): 

    def __init__(self,topk=None,cls_sc_t = 0.7,high_t = 0.5,low_t = 0.3,iou_score_t = 0.7, cnt_t = 5):
        super().__init__()
        self.criterion_logits = nn.BCELoss()
        self.cls_sc_t = cls_sc_t
        self.iou_score_t = iou_score_t
        self.high_t = high_t
        self.low_t = low_t
        self.circle_mask = {}
        self.cnt_t = cnt_t
        self.expend_times = 114514
    def batch_sigmoid_ce_loss(self,inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )

        return loss / hw
    def generate_circle_mask_for_all_points(self,shape, radius):
        height, width = shape
        mask = torch.zeros((height, width, height, width))  # shape (28, 28, 28, 28)
        
        # 遍历每个点 (x, y)，生成以该点为中心的圆形 mask
        for x in range(height):
            for y in range(width):
                # 生成当前点的 mask
                for i in range(height):
                    for j in range(width):
                        if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                            mask[x, y, i, j] = 1
                            
        return mask
    def forward(self,score_map, score, label,gt=None,val=False,topk=None,fin_size=None,mask_circle =False,mask_exp =True,mask_circle_r = 5):
        B, N, h,w = score_map.shape
        
        if (h,w) not in self.circle_mask:
            self.circle_mask[(h,w)] = self.generate_circle_mask_for_all_points((h,w),mask_circle_r).to(score_map.device)
        
        resize_map = score_map.sigmoid()
        if mask_circle and gt is not None:
            resize_map = self.circle_mask[(h,w)].flatten(0,1).unsqueeze(0).detach() * resize_map
        score_xd = score / score.max(dim = 1)[0].unsqueeze(1)


        cls_label = label.nonzero()
        loss = 0
        ret_val = []
        for b,cls_id in cls_label:
            cnt_recycle = 0
            cls_sc,sc_id  = score[b].max(dim = -1)
            cls_sc_xd,sc_id_xd  = score_xd[b].max(dim = -1)

            cls_sc_mask = (cls_sc > self.cls_sc_t) &  (sc_id_xd == cls_id)
            sorted_original_indices = torch.where(cls_sc_mask)[0][cls_sc_xd[cls_sc_mask].argsort(descending=True)] # 先提取总的大于 t 的mask
            # if len(sorted_original_indices) > 0 and len(sorted_original_indices) < 70:
            #     visualize_tensor_list(resize_map[b,sorted_original_indices],name="all_mask.png")
            if topk is not None and len(sorted_original_indices) > topk:
                sorted_original_indices = sorted_original_indices.topk(k=topk)[0]
            # ok 现在应该去挑个最大的
            if not mask_exp and len(sorted_original_indices) > 0:
                # print(sorted_original_indices.shape)
                #if len(sorted_original_indices) == 0
                ans = resize_map[b,sorted_original_indices[0]].unsqueeze(0).unsqueeze(0)
                # print(ans.shape)
                pred = F.interpolate(ans, size=fin_size, mode='bilinear', align_corners=False).max(0)[0][0]
            
            else:
                mask_mask = (sorted_original_indices > -1)

                fin_mask = []
                if len(sorted_original_indices) == 1:
                        max_index = sorted_original_indices[mask_mask][0] if len(fin_mask) > 0 else sorted_original_indices[0]
                        mask_mask[mask_mask.nonzero()[0]] = False
                        ans_mask, mask_mask = self.expand_mask(sorted_original_indices, resize_map, b, gt, max_index, mask_mask,fin_size=fin_size)
                        fin_mask.append(ans_mask.squeeze(0))
                        cnt_recycle += 1
                else:

                    while mask_mask.any(): # 有一个true 就继续
                        max_index = sorted_original_indices[mask_mask][0] if len(fin_mask) > 0 else sorted_original_indices[0]
                        mask_mask[mask_mask.nonzero()[0]] = False
                        ans_mask, mask_mask = self.expand_mask(sorted_original_indices, resize_map, b, gt, max_index, mask_mask,fin_size=fin_size)
                        fin_mask.append(ans_mask.squeeze(0))
                        cnt_recycle += 1
                        if cnt_recycle >= self.cnt_t:
                            break
                    # print(iii)
                if len(fin_mask) == 0:
                    ret_val.append(torch.zeros(fin_size).to(score))
                    continue
                # visualize_tensor_list(fin_mask)
                pred = torch.stack(fin_mask)
                # weights = F.softmax(pred, dim=0)
                pred = pred.max(dim=0)[0] #################################这里改了一下啊
            
            # gt_mask = (gt[b] == cls_id +1)
            # bce_loss = self.criterion_logits(pred, gt_mask.float())
            # iou_loss = (1-batch_iou((pred > 0.7).unsqueeze(0),gt_mask.unsqueeze(0)))
            # print("wsddn seg",1 - iou_loss.item())
            # print(bce_loss.item(),iou_loss[0][0].item(),print(gt_mask.sum() - (pred > 0.7).sum()))
           # print(iou_loss[0][0].item())
            # loss += bce_loss + iou_loss[0]
            # if mask_circle and b == 0:
            #     plt.imshow(pred.detach().cpu(), cmap='gray', vmin=0, vmax=1)
            #     plt.savefig("1.png")
            #     print(1)
            if val:
                ret_val.append(pred)
            if gt is not None:
                gt_mask = (gt[b] == cls_id +1)
                bce_loss = self.criterion_logits(pred,gt_mask.float())
                # bce_loss = get_seg_bce_loss(pred,gt_mask)
                # iou_loss = (1-batch_dice((pred > 0.7).unsqueeze(0),gt_mask.unsqueeze(0)))
                # dice_loss = (1 - ((2 * (pred * gt_mask.float())).sum() + 1) / (pred.sum() + gt_mask.sum() + 1))
                loss += bce_loss
        loss /= cls_label.shape[0]
        if val:
            return loss,ret_val
        else:
            return loss
    def expand_mask(self, sorted_original_indices, resize_map, b, gt, max_index, mask_mask ,fin_size=None):
        
        _high_t = self.high_t
        _low_t = self.low_t
        fin_list = []
        # 先去掉内部的
        # batch_iom()
        # sorted_original_indices[mask_mask]
        
        ori_mask = resize_map[b,max_index].unsqueeze(0) > self.iou_score_t
        queue_mask = resize_map[b,sorted_original_indices] > self.iou_score_t
        iom_scores = batch_iom(queue_mask,ori_mask)
        
        fin_list.append(torch.tensor([max_index]).to(max_index))
        index = sorted_original_indices[((iom_scores >= _low_t) & (iom_scores <= _high_t)).squeeze(-1)]
        mask_mask[(iom_scores > _high_t).nonzero(as_tuple=True)[0]] = False
        mask_mask[((iom_scores >= _low_t) & (iom_scores <= _high_t)).nonzero(as_tuple=True)[0]] = False
        if index.numel() > 0:  # 仅添加非空张量
            fin_list.append(index)
        # fin_list.append(max_index)
            fin_list = [torch.cat(fin_list,dim=0)]
        idx_temp = None
        cir_time = 1
        while True:
            if cir_time >= self.expend_times:
                break
            # oo = resize_map[b,fin_list[0]].max(dim = 0)[0]
            # ori_mask = oo  > 0.7
            if len(ori_mask.shape) == 2:
                ori_mask = ori_mask[None,...]
            # queue_mask = resize_map[b,sorted_original_indices] > 0.7
            # iou_score = (batch_iou(ori_mask,queue_mask) + batch_dice(ori_mask,queue_mask))/2
            iom_scores = batch_iom(queue_mask,ori_mask)
            mask_mask[(iom_scores > _high_t).nonzero(as_tuple=True)[0]] = False
            # index = sorted_original_indices[((iom_scores >= 0.5) & (iom_scores <= 0.8)).squeeze(0)]
            idx = sorted_original_indices[((iom_scores >= _low_t) & (iom_scores <= _high_t)).nonzero(as_tuple=True)[0].unique()]
            idx = idx[(~torch.isin(idx,sorted_original_indices[(~mask_mask).nonzero()[:,0]]))]
            mask_mask[((iom_scores >= _low_t) & (iom_scores <= _high_t)).nonzero(as_tuple=True)[0]] = False
            if (idx_temp is not None and len(idx_temp) == len(idx) and (idx_temp == idx).all()) or len(idx) == 0:
                break
            else:
                fin_list.append(idx)
                fin_list = [torch.cat(fin_list).unique()]
                idx_temp = idx
            cir_time += 1
            # print(cir_time)
            
        # print(resize_map[b,fin_list[0]].unsqueeze(1).shape)
        ans = resize_map[b,fin_list[0]]
        # if len(fin_list[0]) > 0:
        #     visualize_tensor_list(resize_map[b,fin_list[0]])
        #     print("")
        if len(resize_map[b,fin_list[0]].shape) == 2:
            ans = resize_map[b,fin_list[0]].unsqueeze(0).unsqueeze(0)
        else:
            ans = resize_map[b,fin_list[0]].unsqueeze(1)
        if fin_size is not None:
            _size = fin_size
        else:
            _size = gt.shape[1:]
        pred = F.interpolate(ans, size=_size, mode='bilinear', align_corners=False).max(0)[0]
        # pred = ans.max(0)[0]

        return pred,mask_mask
    def update_expand_time(self,change_value):
        self.expend_times = change_value

def batch_iou(pred_masks, gt_masks):
    """
    Calculate the Intersection over Union (IoU) for batches of binary masks.

    Parameters:
    - batch1 (torch.Tensor): A batch of binary masks, shape (N, H, W)
    - batch2 (torch.Tensor): Another batch of binary masks, shape (N, H, W)

    Returns:
    - torch.Tensor: A tensor of IoU scores for each pair in the batches.
    """
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape
    

    pred_masks_exp = pred_masks[:, None, :, :]
    gt_masks_exp = gt_masks[None, :, :, :]
    

    intersection = (pred_masks_exp & gt_masks_exp).float().sum(dim=(2, 3))
    union = (pred_masks_exp | gt_masks_exp).float().sum(dim=(2, 3))
    

    iou = intersection / union
    

    iou[union == 0] = 1.0
    
    return iou
def batch_dice(pred_masks, gt_masks):
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape
    
    # 扩展掩膜维度以便广播
    pred_masks_exp = pred_masks[:, None, :, :]  # 形状变为 (N, 1, H, W)
    gt_masks_exp = gt_masks[None, :, :, :]  # 形状变为 (1, M, H, W)
    
    # 计算交集
    intersection = (pred_masks_exp & gt_masks_exp).float().sum(dim=(2, 3))  # 对高和宽维度求和
    # 计算每个掩膜的总像素
    pred_sum = pred_masks_exp.float().sum(dim=(2, 3))
    gt_sum = gt_masks_exp.float().sum(dim=(2, 3))
    
    # 计算 Dice 系数
    dice = (2 * intersection) / (pred_sum + gt_sum)
    
    # 处理除以零的情况
    dice[pred_sum + gt_sum == 0] = 1.0

    return dice
def batch_iom(pred_masks, gt_masks):
    """
    计算批次中的每个预测掩码和对应的真实掩码的交集除以后者掩码总面积的比值。
    
    Parameters:
    - pred_masks (torch.Tensor): 一批预测掩码，形状为 (N, H, W)
    - gt_masks (torch.Tensor): 一批真实掩码，形状为 (N, H, W)
    
    Returns:
    - torch.Tensor: 每对掩码的交集除以预测掩码总面积的比值。
    """
    # # 将预测掩码和真实掩码转换为浮点型
    # pred_masks = pred_masks.float()
    # gt_masks = gt_masks.float()
    
    # # 计算交集
    # intersection = (pred_masks * gt_masks).sum(dim=(1, 2))
    
    # # 计算预测掩码的总面积
    # pred_area = pred_masks.sum(dim=(1, 2))
    
    # # 避免除以零，设置最小值为1e-6
    # pred_area = torch.clamp(pred_area, min=1e-6)
    
    # # 计算交集除以预测掩码总面积的比值
    # iop = intersection / pred_area
    
    
    
    
    
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape
    

    pred_masks_exp = pred_masks[:, None, :, :]
    gt_masks_exp = gt_masks[None, :, :, :]
    

    intersection = (pred_masks_exp & gt_masks_exp).float().sum(dim=(2, 3))
    gt_area = gt_masks.sum(dim=(1, 2))
    gt_area = torch.clamp(gt_area, min=1e-6)
    

    iop = intersection / gt_area
    

    # iou[union == 0] = 1.0
    

    return iop


class MyGo_Loss3(nn.Module): 

    def __init__(self,topk=None):
        super().__init__()
        self.k = 4
        self.N = self.k
        self.eps = 0.1
        self.max_iter = 100
        self.weights = nn.Parameter(torch.ones(4))
        self.weights.data[2] = 0
        self.weights.data[3] = -1
        self.criterion_logits = nn.BCELoss()

    def kmeans_batch_with_centroids(self,data, k):
        b, n, c = data.shape
        clusters = torch.zeros((b, n), dtype=torch.int)
        centroids = torch.zeros((b, k, c))
        
        for i in range(b):
            batch_data = data[i].cpu().numpy()  # 将张量转换为 numpy 数组，形状为 (n, c)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(batch_data)
            clusters[i] = torch.tensor(kmeans.labels_)
            centroids[i] = torch.tensor(kmeans.cluster_centers_)
    
        return clusters, centroids

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    def ot(self,image_features,text_features):
        M,b,_ = image_features.shape
        _,n_cls,_ = text_features.shape
        image_features =  F.normalize(image_features, dim=2) 
        text_features = F.normalize(text_features, dim=2)


        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M,self.N,b*n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim
        xx=torch.zeros(b*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            return None

        return T
        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)
        

        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_feature_pool @ text_feature_pool.t()
        logits2 = logit_scale * sim_op
        # if self.dataset == "ImageNet":
        #     logits2 = (logits2 + logits)
        return logits2

    def downsample(self,query, score):
    # 隔点采样
        downsampled_query = query[:, ::2, :]
        downsampled_score = score[:, ::2, :]
        return downsampled_query,downsampled_score
    
    def forward(self,query, score, label,gt=None,val=False,topk=None,fin_size=None,shape=None):

        device = query.device  # 获取输入数据所在的设备（CPU or GPU）
        b, n, d = query.shape
        _, _, c = score.shape
        downsampled_query,downsampled_score = self.downsample(query,score)
        # 根据 score 确定每个 query 的类别
        class_assignments = torch.argmax(downsampled_score, dim=-1)  # 假设 score 是 (b, n, c) 维

        # 初始化 clusters 和 centroids 张量
        # clusters = torch.zeros((b, n), dtype=torch.int, device=device)
        # centroids = torch.zeros((b, c, self.k, d), device=device)
        loss = 0
        ret_val = []
        total_n = 0
        for i in range(b):
            for class_idx in range(c):
                if label[i, class_idx] == 0:  # 如果该类不存在，跳过
                    continue

                # 选出属于当前类别的 query
                selected_indices = (class_assignments[i] == class_idx).nonzero(as_tuple=True)[0]
                if selected_indices.shape[0] < self.k:
                    ret_val.append(torch.zeros_like(query[0][:, 0],device=device).unsqueeze(0).reshape(shape[0],shape[1]))
                    continue
                selected_queries = downsampled_query[i][selected_indices]
                all_indices = torch.arange(downsampled_query.shape[1], device=device)
                unselected_indices = all_indices[~torch.isin(all_indices, selected_indices)]
                unselected_queries = downsampled_query[i][unselected_indices]
                backbone_query = unselected_queries.mean(dim=0)
                # s = time.time()
                batch_data = selected_queries.cpu().detach().numpy()  # 将张量转换为 numpy 数组，形状为 (m, d)
                kmeans = KMeans(n_clusters=self.k-1, random_state=0).fit(batch_data)
                # e = time.time()
                # print(f"{selected_indices.shape[0]}loss time:",e-s)
                # clusters[i, selected_indices] = torch.tensor(kmeans.labels_, dtype=torch.int, device=device)
                centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device)
                            # 计算每个聚类的平均 score
                cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int, device=device)
                avg_scores = []
                for k in range(self.k -1):
                    cluster_indices = (cluster_labels == k).nonzero(as_tuple=True)[0]
                    cluster_scores = downsampled_score[i][selected_indices][cluster_indices].mean(dim=0)
                    avg_scores.append(cluster_scores.mean().item())  # 计算平均 score

                # 根据平均 score 对聚类进行排序
                sorted_indices = sorted(range(self.k -1), key=lambda x: avg_scores[x], reverse=True)

                # 将排序结果保存到 ret_val 中
                centroids = centroids[sorted_indices]             
                
                centroids = torch.cat([centroids,backbone_query.unsqueeze(0)])
                
                
                
                x = self.ot(query[0].unsqueeze(0).permute(1,0,2),centroids.unsqueeze(1))
                if x == None:
                    ret_val.append(torch.zeros_like(query[0][:, 0],device=device).unsqueeze(0).reshape(shape[0],shape[1]))
                    continue
                weighted_x = x * self.weights.to(device)
                # 计算加权平均
                pred = weighted_x.sum(dim=-1)
                total_n += 1
                if val:
                    if shape is not None:
                        pred = pred.reshape(shape[0],shape[1])
                    ret_val.append(pred)
                if gt is not None:
                    gt_mask = (gt[i] == class_idx +1)
                    pred = F.interpolate(pred.reshape(1,1,28,28), size=gt.shape[1:], mode='bilinear', align_corners=False).reshape(shape[0],shape[1])
                    bce_loss = self.criterion_logits(pred, gt_mask.float())
                    loss += bce_loss

        # if total_n == 0:
        #     print("233")
        loss /= total_n + 1e-4
        if val:
            return loss,ret_val
        else:
            return loss
            

       



        
