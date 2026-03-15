from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.vit import PatchEmbed
from model.losses import MyGo_Loss2, MyGo_Loss3
from model.maskFormer import Conv2d, ShapeSpec, TransformerEncoderPixelDecoder
from model.query_decoder import DetrTransformerDecoder
from model.transformer_predictor import TransformerPredictor
from model.wsddn_layer import Wsddn_Layer
from utils.wsddnutils import draw_mask, get_atten_mask, get_wsddn_segs
from . import backbone as encoder
from . import decoder
from timm.models.layers import trunc_normal_
import fvcore.nn.weight_init as weight_init
# from model.backbone.clip import load as clip_load, tokenize
from torch.cuda.amp import autocast, GradScaler

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        
        # 定义卷积层，减少通道数并提取特征
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        
        # 定义一个全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 定义全连接层，将特征映射到4个logit
        self.fc = nn.Linear(64, 4)
    
    def forward(self, x):
        # 卷积层1 + 激活函数
        x = F.relu(self.conv1(x))
        
        # 卷积层2 + 激活函数
        x = F.relu(self.conv2(x))
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 展开特征图
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x
class network(nn.Module):
    def __init__(self, args,backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None,get_seg=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.init_momentum = init_momentum
        self.clip = False
        if "clip" in backbone:
            pass
            # self.clip = True
            # self.encoder = clip_load("ViT-B/16",download_root = "pretrained/checkpoints" , aux_layer=aux_layer)[0]
            # VOC_CLASSES = [
            #     "aeroplane", "bicycle", "bird", "boat", "bottle",
            #     "bus", "car", "cat", "chair", "cow", "dining table",
            #     "dog", "horse", "motorbike", "person", "potted plant",
            #     "sheep", "sofa", "train", "tv/monitor"
            # ]
            # templates = [f"a clean origami {cls}" for cls in VOC_CLASSES]
            # text = torch.cat([tokenize(template) for template in templates])
            # with torch.no_grad():
            #     text_features = self.encoder.encode_text(text.to("cuda"))
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            # self.encoder = self.encoder.visual
            # self.decoder_embd_dim = 768
            # # self.classifier.weight = nn.Parameter(text_features[:,:,None,None])
        else:
            self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)
            self.decoder_embd_dim = 768

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 
        self.pooling = F.adaptive_max_pool2d
        if args.decoder ==  "ASSP":
            self.decoder = decoder.ASPP(in_planes=self.decoder_embd_dim, out_planes=self.num_classes,)
        else:
            self.decoder = decoder.LargeFOV(in_planes=self.decoder_embd_dim, out_planes=self.num_classes,)

        
        self.classifier = nn.Conv2d(in_channels=self.decoder_embd_dim, out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        # if "clip" in backbone:
        #     self.classifier.weight = nn.Parameter(text_features[:,:,None,None])
        self.aux_classifier = nn.Conv2d(in_channels=768, out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.query_decoder = Mygo(args,num_classes,self.decoder_embd_dim)
        self.register_buffer("prototypes", torch.randn(args.num_classes - 1,self.decoder_embd_dim))
        # self.query_decoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)
        
        if args.ot_loss:
            self.get_seg = MyGo_Loss3()
            self.cls_token_proj = nn.Linear(in_features=self.decoder_embd_dim,out_features=1)
        else:
            if get_seg is not None:
                self.get_seg = get_seg
            else:
                self.get_seg = MyGo_Loss2(topk=args.wsddn_topk,cls_sc_t=args.cls_sc_t)
            self.cls_token_proj = nn.Linear(self.decoder_embd_dim * 3, 1 * 3)
        
    def init_weigth(self,):
        pass
        ### extract category knowledge 
    @torch.no_grad()
    def update_prototype(self,embeds,label_idx):
        """embeds      : feats from encoder (b,DIM)
           label_idx   : the class index for embeds (b,1)"""
        for feat, label in zip(concat_all_gather(embeds), concat_all_gather(label_idx)):
            for cls_id in label.nonzero():
                self.prototypes[cls_id] = self.prototypes[cls_id]*self.args.proto_m + (1-self.args.proto_m)*feat[cls_id]
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        prototypes = self.prototypes.clone().detach()
        return prototypes
    def get_param_groups(self):

        param_groups = [[], [], [], [],[]] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            # elif  "cls_token" in name:
            #     param_groups[2].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)
        param_groups[2].append(self.cls_token_proj.weight)
        param_groups[2].append(self.cls_token_proj.bias)

 

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.query_decoder.parameters()):
            param_groups[4].append(param)
        



        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x
    
    def bfs_connected_component_mask_8(feature_map, start):
        if not feature_map[start[0], start[1]]:
            return torch.zeros_like(feature_map, dtype=torch.bool)

        height, width = feature_map.shape

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        queue = [start]
        visited = set()
        visited.add(start)

        connected_mask = torch.zeros_like(feature_map, dtype=torch.bool)

        while queue:
            current = queue.pop(0)
            connected_mask[current[0], current[1]] = True
            
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                    if neighbor not in visited and feature_map[neighbor[0], neighbor[1]]:
                        queue.append(neighbor)
                        visited.add(neighbor)

        return connected_mask
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed',}
    def get_atten_masks(self, self_weight,s=None):
        a = self.cam_atten_mask(self_weight,s)
        diag_element = a.diagonal(dim1=1, dim2=2)
        atten = (a < (diag_element + 1e-5).unsqueeze(-1)) & (a > (diag_element - 1e-5).unsqueeze(-1))
        # atten_bfs = [self.bfs_connected_component_mask_8() for i in ]
        
        return ~atten
    
    def cam_atten_mask(self, self_weight,s = 1):
        self_weight = self_weight.mean(1)
        aff_mask = torch.zeros((self.img_size,self.img_size))
        aff_mask = aff_mask.view(1, self.img_size * self.img_size)
        aff_mat = self_weight[:,s:,s:]

        trans_mat = aff_mat / torch.sum(aff_mat, dim=-2, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=-1, keepdim=True)
        # Sinkhorn normalization
        for _ in range(2):
            trans_mat = trans_mat / torch.sum(trans_mat, dim=-2, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=-1, keepdim=True)
        trans_mat = (trans_mat + trans_mat.transpose(-1, -2)) / 2

        for _ in range(1):
            trans_mat = torch.matmul(trans_mat, trans_mat)
        return trans_mat
    
    def func1(self,score,aa,cls_label = None):
        ans = torch.zeros([score.shape[0],28,28]).to(score.device)
        for i,(b_cls_label,sc,a) in enumerate(zip(cls_label,score,aa)):
            for gt_label  in b_cls_label.nonzero().squeeze(-1):
                _,index = sc[:,gt_label].topk(k=1)
                ans[i,a[index].reshape(28,28)] = (gt_label + 1).to(ans)
        return ans
    def forward(self, x, label_idx=None, crops=None,cls_flags_local=None, n_iter=None,cam_only=False,atten_mask = None,cls_label=None,args = None,aug_img = None,mask_circle = False,
                ignore_mask = False):
        if self.clip:
            self.clip = False
            cls_token, _x, x_aux, backbone_weight, penultimate_features = self.encoder.forward_features(x)
        else:
            # cls_token, _x, x_aux, backbone_weight, penultimate_features = self.encoder.forward_features(x,ret_penultimate_features = cam_only)
            cls_token, _x, x_aux, backbone_weight, penultimate_features = self.encoder.forward_features(x)


            # a = self.cam_atten_mask(backbone_weight[-1])
        # a = w
        # a = self.get_atten_masks(backbone_weight.detach(),s = 1)
        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size
        b = x.shape[0]
        if _x.dim() == 3:
            _x4 = self.to_2D(_x, h, w)
        else:
            _x4 = _x
        # if not self.training:
        # return None, _x4
        if x_aux.dim() == 3:
            _x_aux = self.to_2D(x_aux, h, w)
        else:
            _x_aux = x_aux
        if self.clip:
            pass
        #     _feat = self.to_2D(feat, h, w)
        # if cam_only:
        #     cam = F.conv2d(_x4, self.classifier.weight).detach()
        #     cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
        
            with torch.no_grad():
                feature,img = _feat.clone(),x.clone()
        else:
            if aug_img is not None:
                feature,img = _x4.detach(),aug_img.detach()
            else:
                feature,img = _x4.detach(),x.detach()
        # feature,img = _x4,x 
        outputs_seg_masks,scores = self.query_decoder(feature,img,atten_mask=atten_mask)
        # self.query_decoder(x,atten_mask=None)
        # cls_token_logit = cls_token.mean(-1).reshape(b,-1,self.num_classes-1)
        
        if cam_only:
            if self.clip:
                _feat = _feat / _feat.norm(dim=1, keepdim=True)
                cam = F.conv2d(_feat, self.classifier.weight).detach()
                cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
                # cam_wsddn = scores.detach().permute(0,2,1).reshape(scores.shape[0],-1,h,w)
                _,wsddn_segs = self.get_seg(outputs_seg_masks, scores, cls_label.repeat(2,1), fin_size=img.shape[-2:],topk=args.wsddn_topk,val=True,mask_exp = self.args.mask_exp,)
            else:
                cam = F.conv2d(_x4, self.classifier.weight).detach()
                cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
                _x4_expanded = _x.unsqueeze(2)  # [4, 784, 1, 768]
                if args.ema:
                    prototypes_expanded = self.prototypes.unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 768]

                    # 计算余弦相似度，dim=-1 表示在最后一个维度 (768) 上进行相似度计算
                    cos_sim = F.cosine_similarity(_x4_expanded, prototypes_expanded, dim=-1)
                    # pro_cam = F.conv2d(_x4, self.prototypes.unsqueeze(-1).unsqueeze(-1)).detach()
                    pro_cam = torch.abs(self.to_2D(cos_sim, h, w)).detach()
                else:
                    prototypes_expanded = cls_token.unsqueeze(1)  # [1, 1, 20, 768]
                    cos_sim = F.cosine_similarity(_x4_expanded, prototypes_expanded, dim=-1)
                    pro_cam = torch.abs(self.to_2D(cos_sim, h, w)).detach()

                if cls_label is not None:
                    _,wsddn_segs = self.get_seg(outputs_seg_masks, scores, cls_label.repeat(2,1), fin_size=img.shape[-2:],topk=args.wsddn_topk,val=True,mask_circle = mask_circle)
                else:
                    return cls_token,None,cam_aux, cam,pro_cam
            return penultimate_features,cls_token,wsddn_segs,cam_aux, cam,pro_cam
        if self.args.debug:
            feature_map = _x4
            b,dim,_,_ = _x4.shape
            original_image = x
            feature_map_flat = feature_map.view(b, dim, h * w)
            num_classes = 20

            # 处理每个类别
            for i in range(num_classes):
                cls_token_expanded = cls_token[:, i].unsqueeze(-1)  # 选择第 i 个 cls token

                # 计算与每个类别的相似度图
                similarity = F.cosine_similarity(cls_token_expanded, feature_map_flat, dim=1)
                similarity_map = similarity.view(b, h, w)  # (b, h, w)

                # 对相似度图进行插值，将其大小调整为与原图一致
                similarity_map_resized = F.interpolate(similarity_map.unsqueeze(1), size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)

                # 反归一化操作
                mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
                std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
                original_image_denormalized = original_image.cpu() * std + mean

                # 将反归一化后的原图转换为 numpy 形式进行可视化
                original_image_np = original_image_denormalized[0].cpu().detach().numpy()
                original_image_np = np.transpose(original_image_np, (1, 2, 0))  # 转为 (H, W, C) 形式
                original_image_np = np.clip(original_image_np, 0, 255).astype(np.uint8)  # 将像素值裁剪到 [0, 255] 并转换为 uint8 类型

                # 将相似度图转换为 numpy 形式进行可视化
                similarity_map_resized_np = similarity_map_resized[0].cpu().detach().numpy()

                # 归一化相似度图
                similarity_map_resized_np = (similarity_map_resized_np - similarity_map_resized_np.min()) / (similarity_map_resized_np.max() - similarity_map_resized_np.min())
                similarity_map_resized_np[similarity_map_resized_np < 0.90] = 0.0

                fig, ax = plt.subplots(figsize=(8, 8))

                # 显示反归一化后的原始图像作为背景
                ax.imshow(original_image_np, interpolation='nearest')

                # 在原图上叠加相似度图，并设置透明度 alpha
                ax.imshow(similarity_map_resized_np, cmap='viridis', alpha=0.5, interpolation='bilinear')

                # 移除所有边框、坐标轴和刻度
                ax.axis('off')

                # 保存叠加后的图片，没有白边，命名为每个类别的编号
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(f'cls_token_similarity_overlay_class_{i}.png', bbox_inches='tight', pad_inches=0)
                plt.close()
        cls_aux = self.pooling(_x_aux, (1,1))
        cls_aux = self.aux_classifier(cls_aux)
        if self.clip:
            cls_x4 = self.pooling(_feat, (1,1))
        else:
            cls_x4 = self.pooling(_x4, (1,1))
       
        # score2,sim_feat = ot(_x.permute(1,0,2),cls_token.r cls_token.mean(-1).reshape(b,-1,self.num_classes-1)eshape(b,-1,self.num_classes-1,self.decoder_embd_dim),3)
        
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        # cls_x4 = cls_x4 + torch.normal(0.0, 0.08, cls_x4.size()).to(cls_x4)
        cls_aux = cls_aux.view(-1, self.num_classes-1)
        # cls_aux = cls_aux + torch.normal(0.0, 0.08, cls_aux.size()).to(cls_aux)
        # cls_token_logit = self.cls_token_proj(cls_token.reshape(b,3,self.num_classes-1,-1).permute(0, 2, 1, 3).contiguous().view(b,self.num_classes-1,-1)).permute(0,2,1)
        # cls_token_logit  = F.linear(cls_token.reshape(b,3,self.num_classes-1,-1),self.classifier.weight.squeeze().detach())
        # index = torch.arange(20).to(cls_token_logit.device)
        
        # # 使用索引选择最后一个维度的元素
        # cls_token_logit = cls_token_logit[...,index,index]
        # cls_token_logit = self.cls_token_proj(cls_token.reshape(b,3,self.num_classes-1,-1).permute(0, 2, 1, 3).contiguous().view(b,self.num_classes-1,-1)).permute(0,2,1)
        seg = self.decoder(_x4)
        # seg_decoder_output = self.query_decoder(feature,img,atten_mask=atten_mask)

        # weight = self.conv(_x4)
        # if n_iter is not None and n_iter >= self.args.seg_iter:
        #     self.args.pro_m = 0.99
        if n_iter is not None and n_iter >= self.args.update_prototype: 
            prototypes = self.update_prototype(cls_token.contiguous(),cls_label.contiguous()).to(x.device)
        else:
            prototypes = self.prototypes
        return cls_x4, seg, _x4, cls_aux, _x,cls_token,None,None,outputs_seg_masks, scores,prototypes
    def forward_last_layer(self, x):
        x,hw,cls_label = x
        h ,w = hw
        _x,_ = self.encoder.blocks[-1](x)
        if _x.dim() == 3:
            _x4 = self.to_2D(_x[:,_x.shape[1] - h*w:], h, w)
        else:
            _x4 = _x
        cls_x4 = F.adaptive_max_pool2d(_x4, (1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        # F.multilabel_soft_margin_loss(cls_x4, cls_label)
        return cls_x4

def Sinkhorn(K, u, v):
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
def ot(image_features,text_features,N,eps=0.1,logit_scale=4.0):
        M,b,_ = image_features.shape
        _,_,n_cls,_ = text_features.shape
        image_features =  F.normalize(image_features, dim=-1) 
        text_features = F.normalize(text_features, dim=-1)


        sim = torch.einsum('mbd,bncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M,N,b*n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim
        xx=torch.zeros(b*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

        with torch.no_grad():
            KK = torch.exp(-wdist / eps)
            # KK = torch.clamp(KK, min=1e-10, max=1e10)
            T = Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any() or torch.isinf(T).any():
            return None
        sim_feat = T * sim
        sim_op = torch.sum(sim_feat, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,n_cls)
        

        # logit_scale = 4.0
        # logits = logit_scale * image_feature_pool @ text_feature_pool.t()
        logits2 = logit_scale * sim_op
        # if self.dataset == "ImageNet":
        #     logits2 = (logits2 + logits)
        # sim_feat[(sim_feat.sum(-2,keepdim=True) > 0)].sum
        # sim_class = sim_feat.sum(dim=1,keepdim=True) > 0
        # sim_class[sim_class > 0].sum(dim=-1)
        return logits2,T * T.shape[-1]* T.shape[-2]
class Mygo(nn.Module):
    
    def __init__(self,args,num_classes,emd_dim):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = [emd_dim]
        self.query_decoder = DetrTransformerDecoder(self.in_channels[-1],num_layers=2,attn_dropout = 0.0,ffn_dropout = 0.0)
        self.img_size = args.crop_size // 16
        # self.query = nn.Embedding((args.crop_size // 16) ** 2, self.in_channels[-1])
        self.wsddn = Wsddn_Layer(self.in_channels[-1],num_classes - 1,mil = args.mil)
        self.pos_embed = nn.Parameter(torch.zeros(1, (args.crop_size // 16) ** 2, self.in_channels[-1]))
        trunc_normal_(self.pos_embed, std=.02)
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=16, in_chans=3, embed_dim=self.in_channels[-1])
        # self.attn_proj = nn.Linear((args.crop_size // self.encoder.patch_size) ** 2,(args.crop_size // self.encoder.patch_size) ** 2)
        self.ot = args.ot_loss
        # self.logit_scale = args.logit_scale
        self.pooling = F.adaptive_max_pool2d
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.predictor = TransformerPredictor(
            in_channels=self.in_channels[-1],
            mask_classification= True,
            num_classes = num_classes,
            hidden_dim = self.in_channels[-1],
            num_queries = 100,
            nheads = 8,
            dropout=0.1,
            dim_feedforward=2048,
            enc_layers=0,
            dec_layers=6,
            pre_norm=False,
            deep_supervision=False,
            mask_dim=self.in_channels[-1],
            enforce_input_project=False
        )
        if self.ot == False:
            
            self.pixel_decoder = TransformerEncoderPixelDecoder(
                input_shape = {"v12":ShapeSpec(channels=emd_dim, height=28, width=28, stride=16),},
                transformer_dropout = 0.1,
                transformer_nheads = 8,
                transformer_dim_feedforward = 2048,
                transformer_enc_layers = 0,
                transformer_pre_norm = False,
                conv_dim = emd_dim,
                mask_dim = emd_dim,
                norm = "GN",
            )
            self.classifier = nn.Conv2d(in_channels=emd_dim, out_channels=self.num_classes - 1, kernel_size=1, bias=False,)
            self.mask_embed = MLP(emd_dim, emd_dim, emd_dim, 3)
        else:
            # Optimal Transport
            self.N = args.ot_n
            self.num_cls_tokens = self.N* (self.num_classes-1)
            # 初始化 CLS tokens
            self.cls_tokens = nn.Parameter(torch.randn(self.num_cls_tokens, self.in_channels[-1]))
            # self.query_decoder = YourQueryDecoder()  # 替换成你的 query decoder
            pass
    def generate_mask(self, seq_len):
        # 生成一个全为1的mask矩阵
        mask = torch.ones(seq_len, seq_len)
        # 让 CLS tokens 之间互相不进行 attention
        mask[:self.num_cls_tokens, :self.num_cls_tokens] = 0
        return mask
    def forward(self,feature,img,atten_mask=None):
        maskformer = False
        if not maskformer:
            h, w = img.shape[-2] // 16, img.shape[-1] // 16
            b = feature.shape[0]
            atten_mask = get_atten_mask(feature.detach())
            ## ==========query from img===========
            query_from_img = True
            if query_from_img:

                query = self.patch_embed(img)  # patch linear embedding
                patch_pos_embed = self.pos_embed[:, :, :].reshape(1, 28, 28, -1).permute(0, 3, 1, 2)
                patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode="bicubic", align_corners=False)
                patch_pos_embed = patch_pos_embed.reshape(1, -1, h*w).permute(0, 2, 1)

                # add positional encoding to each token 
                query = (query  + patch_pos_embed).permute(1,0,2).contiguous()

            else:
                query = self.query.weight.repeat(b,1,1).permute(1,0,2).contiguous()
            key = feature.flatten(-2).permute(2,0,1)
            # # cls token
            # cls_tokens = self.cls_tokens.unsqueeze(0).expand(b, -1, -1).permute(1,0,2)  # (batch_size, num_cls_tokens, cls_token_dim)
            # query_with_cls = torch.cat((cls_tokens, query), dim=0)
            # attn_mask = self.generate_mask(query_with_cls.size(0)).to(query_with_cls.device)
            
            # out_query = self.query_decoder(query_with_cls , key, key,attn_masks=[attn_mask,None])
            
            # # cross_weight = cross_weight[-1].mean(1)
            # cls_token = out_query[-1, :self.num_cls_tokens,...].permute(1,0,2).reshape(b,self.num_cls_tokens // (self.num_classes-1),(self.num_classes-1),-1)
            # out_query = out_query[-1, self.num_cls_tokens:,...]
            # score2,sim_feat = ot(out_query,cls_token,self.N)
            # out_feat = out_query.permute(1,2,0).reshape(b,-1,h,w)
            # cls_q = self.pooling(out_feat, (1,1))
            # scores = self.classifier(cls_q)
            # scores = self.wsddn(out_query[-1].permute(1,0,2))
            if self.ot == False:
                x_pix, _ = self.pixel_decoder.forward_features(feature)
                out_query = self.query_decoder(query , key, key,attn_masks=[None,None])
                # sim = out_query[-1].permute(1,2,0)
                # if sim.shape[0] == 1:
                #     def cos_sim(x,y = None):
                #         x = F.normalize(x, p=2, dim=1, eps=1e-8)
                #         if y is None:
                #             cos_sim = torch.matmul(x.transpose(1,2), x)
                #         else:
                #             cos_sim = torch.matmul(y.transpose(1,2), x)
                #         return torch.abs(cos_sim)
                #     a = cos_sim(sim,feature.flatten(-2)).reshape(1,-1,28,28)
                #     draw_mask(a.cpu(),"affmask")
                mask_embed = self.mask_embed(out_query[-1].permute(1,0,2))
                outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, x_pix)
                scores = self.wsddn(out_query[-1].permute(1,0,2))
                return outputs_seg_masks,scores
            else:
                return scores, score2,None,out_query.permute(1,0,2),out_feat,sim_feat
        else:
            mask_features, transformer_encoder_features = self.pixel_decoder.forward_features(feature)
            predictions = self.predictor(transformer_encoder_features, mask_features)

            return predictions

        
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
def show_mask(sim_feat):
    images = sim_feat.view(28, 28, 3)

    # 绘制 3 张图片
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(3):
        axes[i].imshow(images[:, :, i], cmap='gray')
        axes[i].set_title(f'Image {i+1}')
        axes[i].axis('off')

    plt.savefig("111.png")