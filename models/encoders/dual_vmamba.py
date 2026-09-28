import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from models.encoders.vmamba import Backbone_VSSM, SaliencyMambaBlock, RAMFBlock
from models.HA import HA
try:
    import clip
except ImportError:
    print("Install CLIP with: pip install git+https://github.com/openai/CLIP.git")

# =====================================================================
# ContextAdapter & CLIPPromptGuide
# =====================================================================
class ContextAdapter(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, num_layers=6):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.ones(embed_dim) * 1e-4)

    def forward(self, text_embeds, image_embeds):
        context_aware_text = self.transformer_decoder(tgt=text_embeds, memory=image_embeds)
        context_aware_text = self.norm_out(context_aware_text)
        return text_embeds + self.gamma * context_aware_text

class CLIPPromptGuide(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        print(">>> Loading CLIP")
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        self.dtype = self.clip_model.dtype
        self.visual = self.clip_model.visual

        
        last_block = self.visual.transformer.resblocks[-1]
        embed_dim = self.visual.transformer.width
        qkv_weight = last_block.attn.in_proj_weight
        qkv_bias = last_block.attn.in_proj_bias
        proj_weight = last_block.attn.out_proj.weight
        proj_bias = last_block.attn.out_proj.bias
            
        self.w_v = qkv_weight[2*embed_dim : 3*embed_dim, :]
        self.b_v = qkv_bias[2*embed_dim : 3*embed_dim]
        self.c_proj_weight = proj_weight
        self.c_proj_bias = proj_bias
        
        self.ln_1 = last_block.ln_1
        self.ln_2 = last_block.ln_2
        self.mlp = last_block.mlp

        self.prompts = [
            "High frequency details, sharp edges, rich texture", 
            "Low frequency, smooth, blurry, flat region",
            "Bright, well-lit region",
            "Dark, shadowed, silhouette region",
            "Clean visual content",
            "Noisy, grainy, distorted artifacts",
            "Informative region, meaningful visual details",  
            "Redundant content, homogeneous background"      
        ]
        
        with torch.no_grad():
            text_tokens = clip.tokenize(self.prompts).to(device)
            self.text_features_static = self.clip_model.encode_text(text_tokens)
            self.text_features_static = self.text_features_static / self.text_features_static.norm(dim=1, keepdim=True)
        
        self.context_adapter = ContextAdapter(embed_dim=512, num_heads=8, num_layers=6)
        self.prompt_weights = nn.Parameter(torch.ones(4))

    def encode_image_dense(self, image):
        x = image.type(self.dtype)
        x = self.visual.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) 
        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  
        layers = self.visual.transformer.resblocks[:-1] 
        x = layers(x)
        
        x = x.permute(1, 0, 2) 
        residual = x
        x = self.ln_1(x)
        v = F.linear(x, self.w_v, self.b_v)
        x = F.linear(v, self.c_proj_weight, self.c_proj_bias)
        x = x + residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual
        x = self.visual.ln_post(x)
        x_patches = x[:, 1:, :] 
        if self.visual.proj is not None:
            x_patches = x_patches @ self.visual.proj
        return x_patches

    def forward(self, image_clips):
        B = image_clips.shape[0]
        with torch.no_grad():
            image_features = self.encode_image_dense(image_clips)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)

        text_static = self.text_features_static.unsqueeze(0).expand(B, -1, -1)
        text_dynamic = self.context_adapter(text_static.float(), image_features.float())
        text_dynamic = text_dynamic / (text_dynamic.norm(dim=-1, keepdim=True) + 1e-6)
        
        logits = torch.matmul(image_features.float(), text_dynamic.transpose(1, 2))
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logits * logit_scale
        
        L = logits.shape[1]
        logits_reshaped = logits.view(B, L, 4, 2)
        probs = logits_reshaped.softmax(dim=-1)
        positive_probs = probs[:, :, :, 0]
        att_weights = F.softmax(self.prompt_weights, dim=0).view(1, 1, 4)
        avg_score = (positive_probs * att_weights).sum(dim=-1, keepdim=True)
        
        grid_size = int(L ** 0.5)
        spatial_score_map = avg_score.permute(0, 2, 1).view(B, 1, grid_size, grid_size)
        return spatial_score_map
# =====================================================================

class RGBXMamba(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=128,
                 pretrained=None,
                 mlp_ratio=0.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[448, 448],
                 patch_size=4,
                 drop_path_rate=0.6,
                 shared_clip_guide=None, 
                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm_r = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        if shared_clip_guide is not None:
            self.clip_guide = shared_clip_guide
        else:
            self.clip_guide = None 
        self.gate_scale = nn.Parameter(torch.zeros(1))

        self.pred_saliency = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, bias=False)
        
        self.saliency_mamba = nn.ModuleList(
            SaliencyMambaBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
                lip = False
            ) for i in range(3)
        )

        self.modality_mamba = nn.ModuleList([
            RAMFBlock(
                hidden_dim=dims * (2 ** i), 
                mlp_ratio=0.0,            
                d_state=4,
                d_conv=3,                   
                expand=2,                   
                drop_path=0.0               
            ) for i in range(4)             
        ])

        self.ha = HA()
        
        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e, image_clips=None): 
        """
        x_rgb: B x C x H x W
        """
        B = x_rgb.shape[0]
        outs_fused = []
        
        
        clip_map = None
        if image_clips is not None and self.clip_guide is not None:
            clip_map = self.clip_guide(image_clips)

        outs_rgb = self.vssm_r(x_rgb) # B x C x H x W
        outs_e = self.vssm_r(x_e) # B x C x H x W
        feat_for_saliency = outs_rgb[3] + outs_e[3]
        saliency = self.pred_saliency(F.interpolate(feat_for_saliency, scale_factor=8, mode='bilinear', align_corners=False))
        
        guide_saliency = torch.nn.Sigmoid()(saliency)
        guide_saliency = self.ha(guide_saliency)

        for i in range(4):
            out_rgb = outs_rgb[i].permute(0, 2, 3, 1).contiguous()
            
            
            out_e = outs_e[i].permute(0, 2, 3, 1).contiguous() 

            
            if clip_map is not None:
                H, W = out_rgb.shape[1], out_rgb.shape[2]
                
                
                s_map = F.interpolate(clip_map, size=(H, W), mode='bilinear', align_corners=False)
                s_map = s_map.permute(0, 2, 3, 1) # [B, H, W, 1]
                
                
                delta = torch.tanh((s_map - 0.5) * self.gate_scale)
                
                w_rgb = 1.0 + delta
                w_e   = 1.0 - delta
                
                out_rgb = out_rgb * w_rgb
                out_e   = out_e * w_e

            if i < 3:
                B,H,W,C = out_rgb.shape
                resized_gt = F.interpolate(saliency, size=(H, W), mode='bilinear', align_corners=False)
                resized_gt  = (resized_gt  >= 0.3).float()
                out_rgb = self.saliency_mamba[i](out_rgb, resized_gt)


            
            
            out_rgb = self.modality_mamba[i](out_rgb, out_e)

            out_rgb = out_rgb.permute(0, 3, 1, 2).contiguous()
            outs_fused.append(out_rgb)        
        return outs_fused, saliency

    def forward(self, x_rgb, x_e, image_clips=None): 
        out, saliency = self.forward_features(x_rgb, x_e, image_clips=image_clips)
        return out, saliency

# class vssm_tiny(RGBXMamba):
#     def __init__(self, fuse_cfg=None, **kwargs):
#         super(vssm_tiny, self).__init__(
#             depths=[2, 2, 9, 2],
#             dims=96,
#             pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
#             mlp_ratio=0.0,
#             downsample_version='v1',
#             drop_path_rate=0.2,
#         )

class vssm_small(RGBXMamba):
    def __init__(self, fuse_cfg=None, shared_clip_guide=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='models/pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
            shared_clip_guide=shared_clip_guide, 
        )

# class vssm_base(RGBXMamba):
#     def __init__(self, fuse_cfg=None, **kwargs):
#         super(vssm_base, self).__init__(
#             depths=[2, 2, 27, 2],
#             dims=128,
#             pretrained='models/pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
#             mlp_ratio=0.0,
#             downsample_version='v1',
#             drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
#         )
