import json
import torch
import numpy as np
from einops.einops import rearrange
import torch.nn as nn
import sys
sys.path.append('../model/')

from encoder import load_encoder
from decoder import load_decoder
from transformers import RobertaTokenizer, RobertaConfig
# from clip import build_model
from layers import FPN, Projector, TransformerDecoder
from fusionnet import FusionNet
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from objective import optimal_transport_dist
            
# SeFusion_with_constractive
class SeCoFusion(nn.Module):
    def __init__(self, config, output)-> None:
        super().__init__()
        self.use_tv_constract = config.use_tv_constract
        self.experts_encoder = FusionNet(output) # 用于将vis、inf 解码并做一次patch contrastive loss
        self.patchembed_img = PatchEmbed(img_size=config.resolution, patch_size=16, in_chans=48, embed_dim=128)
        self.patchembed_tv = PatchEmbed(img_size=config.resolution, patch_size=16, in_chans=96, embed_dim=256)

        # model_config
        fusion_config = json.load(open('../config/fusion.json', 'r'))[config['fusion_model']]
        roberta_config = RobertaConfig.from_dict(fusion_config['roberta_model'])

        # tokenizer
        if config.use_anno:
            self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/ckpts/robert-base/')
        # decoder 
        self.decoder = load_decoder(fusion_config['roberta_model']['model_name'], config=roberta_config)
        
    def forward(self, x_vis, x_inf):
        """
        x_vis: [1,3,256,256];
        x_inf: [1,1,256,256]
        """

        cat_in, vis_enc, inf_enc = self.experts_encoder(x_vis, x_inf)
        vi_emb = self.patchembed_img(vis_enc)  # [B,256,128]
        inf_emb = self.patchembed_img(inf_enc) # [B,256,128]
        if self.use_tv_constract:
            img_emb = self.patchembed_tv(cat_in)
        # 计算patch-align loss
        pa_loss = optimal_transport_dist(vi_emb, inf_emb)

        

class PatchEmbed(nn.Module):
    '''
    将vis，inf patchify：
    img: [B, C, H, W] --> [B, C, H*W]
    parms:
        x_in: input to be patchified
        patch_size: window size
        stride: soso
    '''
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        print(f"x :{x.shape}")
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

if __name__ == '__main__':
    # test patchEmbed
    x = torch.randn(1,48,256,256)
    patchify = PatchEmbed(img_size=256, patch_size=16, in_chans=48, embed_dim=128)

    print(patchify(x).shape)

        
        





        
