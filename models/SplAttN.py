from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import time
from models.model_utils import *
from models.tinyvit.tiny_vit import tiny_vit_5m_224
from models.EdgeConv import DGCNN_Grouper
from models.Transformer import EncoderBlock
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=256, num_tokens=128, num_heads=4, depth=3):
        super(FeatureExtractor, self).__init__()
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        
        self.grouper = DGCNN_Grouper()
        
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, out_dim, 1)
        )
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(out_dim, out_dim, 1)
        )
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                dim=out_dim, 
                num_heads=num_heads, 
                mlp_ratio=4.0,
                drop=0.0,
                attn_drop=0.0
            ) for _ in range(depth)
        ])
        
        self.proj = nn.Conv1d(out_dim, out_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N)
        
        Returns:
            tokens: (B, out_dim, num_tokens)
            global_feat: (B, out_dim, 1)
        """
        coor, f = self.grouper(point_cloud)  # coor: (B, 3, 128), f: (B, 128, 128)
        
        pos = self.pos_embed(coor)  # (B, out_dim, 128)
        x = self.input_proj(f)  # (B, out_dim, 128)
        x = x + pos  # (B, out_dim, 128)
        x = x.transpose(1, 2).contiguous()  # (B, 128, out_dim)
        
        for blk in self.encoder_blocks:
            x = blk(x)  # (B, 128, out_dim)
        
        tokens = x.transpose(1, 2).contiguous()  # (B, out_dim, 128)
        tokens = self.proj(tokens)  # (B, out_dim, 128)
        global_feat = torch.max(tokens, dim=2, keepdim=True)[0]  # (B, out_dim, 1)
        
        return tokens, global_feat


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class SDG(nn.Module):
    def __init__(self, channel=128,ratio=1,hidden_dim = 512,dataset='ShapeNet'):
        super(SDG, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim

        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = self_attention(channel*2,hidden_dim,dropout=0.0,nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0,nhead=8)

        self.decoder1 = SDG_Decoder(hidden_dim,channel,ratio) if dataset == 'ShapeNet' else self_attention(hidden_dim, channel * ratio, dropout=0.0,nhead=8)

        self.decoder2 = SDG_Decoder(hidden_dim,channel,ratio) if dataset == 'ShapeNet' else self_attention(hidden_dim, channel * ratio, dropout=0.0,nhead=8)

        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio*2, channel*ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=256,layer_dims=[256,hidden_dim])
        self.sigma = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.cd_distance = chamfer_3DDist()


    def forward(self, local_feat, coarse,f_g,partial):
        batch_size, _, N = coarse.size()
        F = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(self.conv_11(f_g)))
        F = torch.cat([F, f_g.repeat(1, 1, F.shape[-1])], dim=1)

        # Structure Analysis
        half_cd = self.cd_distance(coarse.transpose(1, 2).contiguous(), partial.transpose(1, 2).contiguous())[
                      0] / self.sigma
        embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        F_Q = self.sa1(F,embd)
        F_Q_ = self.decoder1(F_Q)

        # Similarity Alignment
        local_feat = self.mlpp(local_feat)
        F_H = self.cross1(F_Q,local_feat)
        F_H_ = self.decoder2(F_H)

        F_L = self.conv_delta(self.conv_ps(torch.cat([F_Q_,F_H_],1)).reshape(batch_size,-1,N*self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1,1,self.ratio) + O_L

        return fine

class TinyViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None, out_features=128):
        super(TinyViTFeatureExtractor, self).__init__()
        self.backbone = tiny_vit_5m_224(pretrained=False, num_classes=0)
        self.tinyvit_out_dim = 320
        
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('head.') and 'attention_bias_idxs' not in k}
            self.backbone.load_state_dict(state_dict, strict=False)
        
        self.proj = nn.Linear(self.tinyvit_out_dim, out_features)
        
    def forward(self, x):
        """
        Args:
            x: [B*num_views, 3, H, W]
        Returns:
            features: [B*num_views, out_features]
        """
        feat = self.backbone.forward_features(x)  # [B*V, 320]
        feat = self.proj(feat)  # [B*V, out_features]
        return feat


class SVFNet(nn.Module):
    def __init__(self, cfg):
        super(SVFNet, self).__init__()
        self.channel = 64
        self.num_views = 3
        self.num_tokens = 128
        
        self.point_feature_extractor = FeatureExtractor(out_dim=256, num_tokens=128, num_heads=4, depth=3)
        self.view_distance = cfg.NETWORK.view_distance
        self.relu = nn.GELU()
        
        self.sa = self_attention(self.channel*8, self.channel*8, dropout=0.0)
        
        self.cross_attn_3d_2d = cross_attention(d_model=256, d_model_out=256, nhead=4, dropout=0.0)
        
        self.posmlp = MLP_CONV(3, [64, 256])
        
        tinyvit_pretrained = getattr(cfg.NETWORK, 'tinyvit_pretrained', 
                                     'models/tinyvit/tiny_vit_5m_22kto1k_distill.pth')
        self.img_feature_extractor = TinyViTFeatureExtractor(
            pretrained_path=tinyvit_pretrained,
            out_features=256
        )
        
        self.img_proj = nn.Linear(256, 256)
        
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(512 + self.channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(512, self.channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(512 + self.channel, self.channel*8, kernel_size=1)

    def forward(self, points, depth):
        """
        Args:
            points: [B, 3, N]
            depth: [B*num_views, 3, H, W]
        Returns:
            f_g: [B, 512, 1]
            coarse: [B, 3, 256]
        """
        batch_size, _, N = points.size()
        
        f_p_tokens, f_p_global = self.point_feature_extractor(points)  # (B, 256, 128), (B, 256, 1)
        f_v_flat = self.img_feature_extractor(depth)  # [B*3, 256]
        f_v = f_v_flat.view(batch_size, self.num_views, -1)  # [B, 3, 256]
        f_v = self.img_proj(f_v)  # [B, 3, 256]
        
        view_point = torch.tensor(
            [0, 0, -self.view_distance, -self.view_distance, 0, 0, 0, self.view_distance, 0],
            dtype=torch.float32
        ).view(-1, 3, 3).permute(0, 2, 1).expand(batch_size, 3, 3).to(depth.device)
        view_pos = self.posmlp(view_point)  # [B, 256, 3]
        
        f_v_t = f_v.transpose(1, 2).contiguous()  # [B, 256, 3]
        f_v_with_pos = f_v_t + view_pos  # [B, 256, 3]
        f_fused = self.cross_attn_3d_2d(f_p_tokens, f_v_with_pos)  # [B, 256, 128]
        
        f_3d_global = torch.max(f_fused, dim=2, keepdim=True)[0]  # [B, 256, 1]
        f_g = torch.cat([f_p_global, f_3d_global], dim=1)  # [B, 512, 1]
        
        x = self.relu(self.ps(f_g))  # [B, 64, 128]
        x = self.relu(self.ps_refuse(torch.cat([x, f_g.repeat(1, 1, x.size(2))], 1)))  # [B, 512, 128]
        x2_d = self.sa(x).reshape(batch_size, self.channel*4, N//8)  # [B, 256, 256]
        
        coarse = self.conv_out(self.relu(self.conv_out1(
            torch.cat([x2_d, f_g.repeat(1, 1, x2_d.size(2))], 1)
        )))  # [B, 3, 256]

        return f_g, coarse

class local_encoder(nn.Module):
    def __init__(self, cfg):
        super(local_encoder, self).__init__()
        self.local_number = cfg.NETWORK.local_points
        
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.gcn_2 = EdgeConv(64, 256, 8)
        
        self.sa_module = self_attention(d_model=256, d_model_out=256, nhead=4, dropout=0.0)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(256 + 256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        self.proj = nn.Conv1d(256, 256, kernel_size=1)

    def forward(self, input):
        """
        Args:
            input: [B, 3, N]
        Returns:
            x: [B, 256, local_number]
        """
        x1 = self.gcn_1(input)  # [B, 64, 2048]
        idx = furthest_point_sample(input.transpose(1, 2).contiguous(), self.local_number)  # [B, 512]
        x1 = gather_points(x1, idx)  # [B, 64, 512]
        x2 = self.gcn_2(x1)  # [B, 256, 512]
        x2_attn = self.sa_module(x2)  # [B, 256, 512]
        x_fused = torch.cat([x2, x2_attn], dim=1)  # [B, 512, 512]
        x_fused = self.fusion(x_fused)  # [B, 256, 512]
        x = self.proj(x_fused)  # [B, 256, 512]
        
        return x

class Model(nn.Module):
    """
    Main model for point cloud completion.
    """
    # Internal signature for verification (do not modify)
    _VERSION = "1.0.0"
    _SIGNATURE = 0x53504C4154  # Internal identifier
    
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.encoder = SVFNet(cfg)
        self.localencoder = local_encoder(cfg)
        self.merge_points = cfg.NETWORK.merge_points
        self.refine1 = SDG(ratio=cfg.NETWORK.step1,hidden_dim=768,dataset=cfg.DATASET.TEST_DATASET)
        self.refine2 = SDG(ratio=cfg.NETWORK.step2,hidden_dim=512,dataset=cfg.DATASET.TEST_DATASET)

    def forward(self, partial,depth):
        partial = partial.transpose(1,2).contiguous()  # (B, 3, N)
        feat_g, coarse = self.encoder(partial,depth)  # (B, 512, 1), (B, 3, 256)
        local_feat = self.localencoder(partial)  # (B, 256, 512)

        coarse_merge = torch.cat([partial,coarse],dim=2)  # (B, 3, 2304)
        coarse_merge = gather_points(coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points))  # (B, 3, 512)

        fine1 = self.refine1(local_feat, coarse_merge, feat_g,partial)  # (B, 3, 2048)
        fine2 = self.refine2(local_feat, fine1, feat_g,partial)  # (B, 3, 16384)

        return (coarse.transpose(1, 2).contiguous(),fine1.transpose(1, 2).contiguous(),fine2.transpose(1, 2).contiguous())