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

class SDiFuion(nn.Module):
    def __init__(self, config):#
        super().__init__()
        ## 输入是rgb and infra-red
        self.experts = {'rgb' : 3,
                        'infred' : 1}
         
        # encoder 
        ## rgb: [1,3,480,480] infred: [1,1,224,224]
        self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/ckpts/robert-base/')
        self.expert_encoder = load_encoder(config['vit_model'], experts=self.experts, image_resolution=config['image_resolution'])
        self.prepare_to_train(config['freeze'])
        self.get_ignored_modules(config['freeze'])
        # Decoder
        self.decoder = TransformerDecoder(
            num_layers=config.num_layers,
            d_model = config.d_model,
            nhead=config.num_head,
            dim_ffn=config.dim_ffn,
            dropout=config.dropout,
            return_intermediate=config.intermediate
        )
        # # Projector
        # self.proj = Projector(
        #     config.word_dim,
        #     config.d_model // 2,
        #     3
        # )

    def prepare_to_train(self, mode='none'):
            for name, params in self.named_parameters():
                if mode == 'freeze_lang':
                    if 'encoder.layer' in name and all(key not in name for key in ['1.self', '1.output', 'adaptor']):
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                elif mode == 'freeze_vision':
                    if 'transformer.resblocks' in name and 'adaptor' not in name:
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                elif mode == 'freeze_lang_vision':
                    if 'encoder.layer' in name and all(key not in name for key in ['1.self', '1.output', 'adaptor']):
                        params.requires_grad = False
                    elif 'transformer.resblocks' in name and 'adaptor' not in name:
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                else:
                    params.requires_grad = True

    def get_ignored_modules(self, mode='none'):
        ignored_modules = []
        if mode == 'freeze_lang':
            for l in range(len(self.text_decoder.roberta.encoder.layer)):
                ignored_modules += [
                    self.text_decoder.roberta.encoder.layer[l][0].attention,
                    self.text_decoder.roberta.encoder.layer[l][0].intermediate,
                    self.text_decoder.roberta.encoder.layer[l][0].output,
                ]
        elif mode == 'freeze_vision':
            for l in range(len(self.expert_encoder.transformer.resblocks)):
                ignored_modules += [
                    self.expert_encoder.transformer.resblocks[l][0].attn,
                    self.expert_encoder.transformer.resblocks[l][0].mlp,
                    self.expert_encoder.transformer.resblocks[l][0].ln_1,
                    self.expert_encoder.transformer.resblocks[l][0].ln_2,
                ]
        elif mode == 'freeze_lang_vision':
            for l in range(len(self.text_decoder.roberta.encoder.layer)):
                ignored_modules += [
                    self.text_decoder.roberta.encoder.layer[l][0].attention,
                    self.text_decoder.roberta.encoder.layer[l][0].intermediate,
                    self.text_decoder.roberta.encoder.layer[l][0].output,
                ]
            for l in range(len(self.expert_encoder.transformer.resblocks)):
                ignored_modules += [
                    self.expert_encoder.transformer.resblocks[l][0].attn,
                    self.expert_encoder.transformer.resblocks[l][0].mlp,
                    self.expert_encoder.transformer.resblocks[l][0].ln_1,
                    self.expert_encoder.transformer.resblocks[l][0].ln_2,
                ]
        else:
            ignored_modules = None
        return ignored_modules
    
    def forward(self, experts, caption=None, train=True, prefix=''):
        device = experts['rgb'].device
        if train:
            experts_train = self.expert_encoder(experts)
            experts_train = rearrange(experts_train, 'l b d -> b l d')  # batch_size, num_latents, output_dim

            caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=30, return_tensors="pt").to(device)
            answer_targets = caption.input_ids.masked_fill(caption.input_ids == self.tokenizer.pad_token_id, -100)
            print(f"====Test experts_train : {experts_train}]")

            # decoder
            
# SeFusion_with_constractive
class SeCoFusion(nn.Module):
    def __init__(self, config, output)-> None:
        super().__init__()
        self.use_tv_constract = config.use_tv_constract
        self.siam_encoder = FusionNet(output) # 用于将vis、inf 解码并做一次patch contrastive loss
        self.patchembed_img = PatchEmbed(img_size=config.resolution, patch_size=16, in_chans=48, embed_dim=128)
        self.patchembed_tv = PatchEmbed(img_size=config.resolution, patch_size=16, in_chans=96, embed_dim=256)


    def forward(self, x_vis, x_inf):
        """
        x_vis: [1,3,256,256];
        x_inf: [1,1,256,256]
        """

        cat_in, vis_enc, inf_enc = self.siam_encoder(x_vis, x_inf)
        vi_emb = self.patchembed_img(vis_enc)
        inf_emb = self.patchembed_img(inf_enc)
        if self.use_tv_constract:
            img_emb = self.patchembed_tv(cat_in)
        

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
    x = torch.randn(1,3,256,256)
    patchify = PatchEmbed(img_size=256, patch_size=16, in_chans=3, embed_dim=48)

    print(patchify(x).shape)

        
        





        
