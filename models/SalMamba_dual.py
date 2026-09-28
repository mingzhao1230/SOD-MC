import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



from .encoders.dual_vmamba import vssm_small as backbone, CLIPPromptGuide
from .decoders.MambaDecoder import MambaDecoder


def zero_module(module):
    """Zero-initialize all parameters in a module."""
    for p in module.parameters():
        p.detach().zero_()
    return module

class ZeroConv(nn.Module):
    """A collection of zero-initialized convolution layers."""
    def __init__(self, channels):
        super(ZeroConv, self).__init__()
        self.convs = nn.ModuleList()
        for ch in channels:
            
            self.convs.append(zero_module(nn.Conv2d(ch, ch, 1)))

    def forward(self, x_list):
        out_list = []
        for x, conv in zip(x_list, self.convs):
            out_list.append(conv(x))
        return out_list

def modality_drop(x_rgb, x_depth):
    """
    Simulate missing modalities by masking RGB, depth, or neither.
    """
    
    
    prob = np.array((1 / 3, 1 / 3, 1 / 3))
    
    
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    
    p = []
    for i in range(x_rgb.shape[0]):
        index = np.random.choice([0, 1, 2], size=1, replace=True, p=prob)[0]
        p.append(modality_combination[index])
    
    p = torch.tensor(p).float().to(x_rgb.device)
    p = p.unsqueeze(2).unsqueeze(3).unsqueeze(4) # Shape: [B, 2, 1, 1, 1]

    
    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]
    
    return x_rgb, x_depth
# ===============================================

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        
        print(">>> [Model Init] Initializing TWO Separate CLIP Guides (Main & Ctrl) <<<")
        self.clip_main = CLIPPromptGuide(device='cuda')
        self.clip_ctrl = CLIPPromptGuide(device='cuda')

        
        
        self.backbone = backbone(shared_clip_guide=self.clip_main)
        self.channels = [96, 192, 384, 768]
        
        
        
        self.backbone_ctrl = backbone(shared_clip_guide=self.clip_ctrl) 
        self.zero_conv = ZeroConv(self.channels) 
        
        
        self.decoder = MambaDecoder(img_size=[448, 448],
                                    in_channels=self.channels, 
                                    num_classes=1, 
                                    depths=[4, 4, 4, 4],
                                    embed_dim=self.channels[0], 
                                    deep_supervision=False)
                                    
        
        self.register_buffer('current_epoch', torch.tensor(0))

    def forward(self, rgb, modal_x, epoch=None, change_epoch=50):
        
        if epoch is not None:
            self.current_epoch.fill_(epoch)
        cur_epoch = self.current_epoch.item()
        
        
        is_stage2 = cur_epoch > change_epoch
        
        
        feat_teacher = None
        rgb_clean, modal_x_clean = rgb, modal_x 
        
        
        
        if self.training and is_stage2:
            
            
            with torch.no_grad():
                image_clips_clean = F.interpolate(rgb_clean, size=(224, 224), mode='bilinear', align_corners=False)
                feat_teacher, _ = self.backbone(rgb_clean, modal_x_clean, image_clips=image_clips_clean)
            
            
            rgb, modal_x = modality_drop(rgb, modal_x)

        orisize = rgb.shape
        
        
        
        image_clips = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)

        
        
        
        x_main, saliency_main = self.backbone(rgb, modal_x, image_clips=image_clips)
        
        
        
        
        
        
        x_ctrl, _ = self.backbone_ctrl(rgb, modal_x, image_clips=image_clips)
        x_ctrl_out = self.zero_conv(x_ctrl)
            
        
        x_final = []
        for feat_m, feat_c in zip(x_main, x_ctrl_out):
            x_final.append(feat_m + feat_c)

        
        out = self.decoder.forward(x_final)
        
        
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        saliency = F.interpolate(saliency_main, size=orisize[2:], mode='bilinear', align_corners=False)
        
        
        return out, saliency, feat_teacher, x_final
    
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randn(2, 3, 448, 448).to(device)
    depth = torch.randn(2, 3, 448, 448).to(device)
    
    out, _, _, _ = model(image, depth)
    print("Output shape:", out.shape)
