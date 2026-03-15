
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
def get_atten_mask(cams):
    # Flatten the feature maps from the cams
    # gt = cams.flatten(-2)  # Shape: (batch_size, num_patches, flattened_size)
    
    # # Compute the patch comparison mask
    # ret = gt.unsqueeze(1).repeat(1, gt.shape[1], 1)
    # patch_mask = ret != gt.unsqueeze(-1)  # Shape: (batch_size, num_patches, num_patches)
    
    # Add an extra dimension for the cls token
    # batch_size, num_patches, _ = patch_mask.shape
    # full_mask = torch.zeros((batch_size, num_patches + 1, num_patches + 1), dtype=torch.bool).to(patch_mask)
    # full_mask[:, 1:, 1:] = patch_mask
    
    inputs = cams
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)
    a = inputs_cos.reshape(inputs_cos.shape[0],-1,h,w)
    if False:
        a = inputs_cos.reshape(inputs_cos.shape[0],-1,28,28)
        draw_mask(a > 0.99,"affmask")

    return (a > 0.99).flatten(-2)
def visualize_tensor_list(tensor_list,name = "visualize_tensor_list.png"):
    """
    可视化包含二维张量的列表。
    
    参数:
    - tensor_list: list of 2D tensors，每个 tensor 形状应为 (H, W)
    """
    # 计算行列数，方便布局
    num_tensors = len(tensor_list)
    cols = min(5, num_tensors)  # 每行最多显示 5 个
    rows = (num_tensors + cols - 1) // cols  # 根据总数计算行数

    plt.figure(figsize=(cols * 3, rows * 3))  # 设置图表大小

    for i, tensor in enumerate(tensor_list):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(tensor.detach().cpu().numpy() > 0.1, cmap='gray')  # 将 tensor 转为 numpy 格式
        plt.axis('off')
        plt.title(f"Tensor {i+1}")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def draw_mask(seg_mask,name):
    # import torch
    # import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import math
    # 示例相似度图张量
    similarity_maps = seg_mask
    def highlight_point(image, x, y, highlight_color=(1, 0, 0)):
        """
        在灰度图像的 (x, y) 位置添加红色高亮。
        
        参数：
            image (torch.Tensor): 2D灰度图像，形状为 (28, 28)。
            x (int): 要高亮的x坐标（行）。
            y (int): 要高亮的y坐标（列）。
            highlight_color (tuple): 高亮颜色，默认为红色 (1, 0, 0)。
        
        返回：
            torch.Tensor: RGB图像，形状为 (3, 28, 28)。
        """
        # 归一化灰度图到 [0,1]
        img_min = image.min()
        img_max = image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-5)
        
        # 转换为RGB
        img_rgb = torch.stack([img_norm, img_norm, img_norm], dim=0)  # (3, 28, 28)
        
        # 添加红色高亮
        if 0 <= x < 28 and 0 <= y < 28:
            img_rgb[0, x, y] = highlight_color[0]  # R
            img_rgb[1, x, y] = highlight_color[1]  # G
            img_rgb[2, x, y] = highlight_color[2]  # B
        
        return img_rgb
    def get_highlight_position(index, image_size=28):
        """
        根据图像索引计算要高亮的位置 (x, y)。
        
        参数：
            index (int): 图像索引。
            image_size (int): 图像的尺寸（默认28）。
        
        返回：
            tuple: (x, y) 坐标。
        """
        x = index // image_size
        y = index % image_size
        # 确保坐标不超出范围
        x = min(x, image_size - 1)
        y = min(y, image_size - 1)
        return x, y
    # 移除批次维度
    similarity_maps = similarity_maps.squeeze(0)  # 形状变为 (768, 28, 28)

    # 获取图像总数
    total_images = similarity_maps.shape[0]

    # 初始化一个列表存储处理后的RGB图像
    processed_images = []

    for idx in range(total_images):
        img = similarity_maps[idx]
        x, y = get_highlight_position(idx, image_size=28)
        img_rgb = highlight_point(img, x, y)  # (3, 28, 28)
        processed_images.append(img_rgb)

    # 转换为张量
    processed_images = torch.stack(processed_images)  # (768, 3, 28, 28)
    # 定义每个网格的大小
    images_per_grid = 100
    grid_size = 10  # 10x10网格
    num_grids = math.ceil(total_images / images_per_grid)

    for i in range(num_grids):
        start_idx = i * images_per_grid
        end_idx = min((i + 1) * images_per_grid, total_images)
        batch = processed_images[start_idx:end_idx]
        
        # 如果当前批次少于images_per_grid，则补零
        if batch.shape[0] < images_per_grid:
            padding = images_per_grid - batch.shape[0]
            padding_tensor = torch.zeros(padding, 3, 28, 28)
            batch = torch.cat([batch, padding_tensor], dim=0)
        
        # 创建网格
        grid = make_grid(batch, nrow=grid_size, padding=2, normalize=False)
        
        # 转换为numpy数组并绘制
        np_grid = grid.numpy().transpose((1, 2, 0))  # (H, W, C)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(np_grid)
        plt.axis('off')
        plt.title(f'Similarity Grid {i+1}')
        
        # 保存图片
        plt.savefig(f'similarity_grid_{i+1}.png', bbox_inches='tight')
        plt.close()

        


def get_wsddn_segs(score_map, score, label,cls_sc_t = 0.8,fin_size=(448,448),iou_score_t = 0.7,topk = 300):
    B, N, h,w = score_map.shape
    resize_map = score_map.sigmoid()
    # 沿着最后一个维度找到最小值和最大值  
    min_values, _ = torch.min(score, dim=1, keepdim=True)  
    max_values, _ = torch.max(score, dim=1, keepdim=True)  
    
    # 防止除以零（如果最大值和最小值相同）  
    eps = 1e-8  # 一个小的正数  
    max_values = torch.where(max_values == min_values, max_values + eps, max_values)  
    
    # 缩放到0-1之间  
    score = (score - min_values) / (max_values - min_values)  
    # score = score / score.max(dim = 1)[0].unsqueeze(1)


    cls_label = label.nonzero()
    loss = 0
    ret_val = []
    for b,cls_id in cls_label:
        cls_sc = score[b,:,cls_id]
        cls_sc_mask = cls_sc > cls_sc_t
        sorted_original_indices = torch.where(cls_sc_mask)[0][cls_sc[cls_sc_mask].argsort(descending=True)]
        if len(sorted_original_indices) > topk:
            sorted_original_indices = sorted_original_indices.topk(k=topk)[0]
        # print(cls_sc_mask.sum())
        mask_mask = (sorted_original_indices > -1)
        mask_mask[0] = False
        fin_mask = []
        if len(sorted_original_indices) == 1:
            max_index = sorted_original_indices[0]
            ans_mask, a = expand_mask(sorted_original_indices, resize_map,b,fin_size,max_index,iou_score_t)
            mask_mask = ~torch.isin(sorted_original_indices, a) & mask_mask
            fin_mask.append(ans_mask.squeeze(0))
        else:
            iii = 0
            while mask_mask.any(): # 有一个true 就继续
                
                max_index = sorted_original_indices[mask_mask][0] if len(fin_mask) > 0 else sorted_original_indices[0]
                ans_mask,a = expand_mask(sorted_original_indices, resize_map,b,fin_size,max_index,iou_score_t)
                mask_mask = ~torch.isin(sorted_original_indices, a) & mask_mask
                fin_mask.append(ans_mask.squeeze(0))
                iii += 1
            # print(iii)
        
        pred = torch.stack(fin_mask)
        weights = F.softmax(pred, dim=0)
        pred = (weights * pred).sum(dim=0)
    

        ret_val.append(pred.detach())

    return ret_val

def expand_mask(sorted_original_indices, resize_map,b,fin_size,max_index,iou_score_t):
    fin_list = []
    # max_index = sorted_original_indices[0]
    ori_mask = resize_map[b,max_index].unsqueeze(0) > 0.7
    queue_mask = resize_map[b,sorted_original_indices] > 0.7
    iou_score = (batch_iou(ori_mask,queue_mask) + batch_dice(ori_mask,queue_mask))/2
    fin_list.append(sorted_original_indices[(iou_score > iou_score_t).squeeze(0)])
    # if (iou_score > self.iou_score_t).sum() == sorted_original_indices.shape[0]:
    #     print("一次ok")
    # else:
    #     print("多次ok")
    fin_list = [torch.cat(fin_list)]
    while True:
        ori_mask = resize_map[b,fin_list[0]] > 0.7
        queue_mask = resize_map[b,sorted_original_indices] > 0.7
        iou_score = (batch_iou(ori_mask,queue_mask) + batch_dice(ori_mask,queue_mask))/2
        idx = sorted_original_indices[(iou_score > iou_score_t).nonzero(as_tuple=True)[1].unique()]
        if fin_list[0].shape[0] == idx.shape[0]:
            break
        else:
            fin_list.append(idx)
            fin_list = [torch.cat(fin_list).unique()]
    # print(resize_map[b,fin_list[0]].unsqueeze(1).shape)
    pred = F.interpolate(resize_map[b,fin_list[0]].unsqueeze(1), size=fin_size, mode='bilinear', align_corners=False).mean(0)
    return pred,fin_list[0]

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
    
    # 扩展掩膜维度以便广播
    pred_masks_exp = pred_masks[:, None, :, :]  # 形状变为 (N, 1, H, W)
    gt_masks_exp = gt_masks[None, :, :, :]  # 形状变为 (1, M, H, W)
    
    # 计算交集和并集
    intersection = (pred_masks_exp & gt_masks_exp).float().sum(dim=(2, 3))  # 对高和宽维度求和
    union = (pred_masks_exp | gt_masks_exp).float().sum(dim=(2, 3))
    
    # 计算 IoU
    iou = intersection / union
    
    # 处理除以零的情况
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