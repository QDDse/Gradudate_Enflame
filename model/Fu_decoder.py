import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../model/')
from encoder import load_encoder
from decoder import load_decoder
from einops.einops import rearrange

from transformers import RobertaTokenizer, RobertaConfig

# class SDiFuion(nn.Module):
#     def __init__(self, config):#
#         super().__init__()
#         ## 输入是rgb and infra-red
#         self.experts = {'rgb' : 3,
#                         'infred' : 1}
         
#         # encoder 
#         ## rgb: [1,3,480,480] infred: [1,1,224,224]
#         self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/ckpts/robert-base/')
#         self.expert_encoder = load_encoder(config['vit_model'], experts=self.experts, image_resolution=config['image_resolution'])
#         self.prepare_to_train(config['freeze'])
#         self.get_ignored_modules(config['freeze'])
#         # Decoder
#         self.decoder = TransformerDecoder(
#             num_layers=config.num_layers,
#             d_model = config.d_model,
#             nhead=config.num_head,
#             dim_ffn=config.dim_ffn,
#             dropout=config.dropout,
#             return_intermediate=config.intermediate
#         )
#         # # Projector
#         # self.proj = Projector(
#         #     config.word_dim,
#         #     config.d_model // 2,
#         #     3
#         # )

#     def prepare_to_train(self, mode='none'):
#             for name, params in self.named_parameters():
#                 if mode == 'freeze_lang':
#                     if 'encoder.layer' in name and all(key not in name for key in ['1.self', '1.output', 'adaptor']):
#                         params.requires_grad = False
#                     else:
#                         params.requires_grad = True
#                 elif mode == 'freeze_vision':
#                     if 'transformer.resblocks' in name and 'adaptor' not in name:
#                         params.requires_grad = False
#                     else:
#                         params.requires_grad = True
#                 elif mode == 'freeze_lang_vision':
#                     if 'encoder.layer' in name and all(key not in name for key in ['1.self', '1.output', 'adaptor']):
#                         params.requires_grad = False
#                     elif 'transformer.resblocks' in name and 'adaptor' not in name:
#                         params.requires_grad = False
#                     else:
#                         params.requires_grad = True
#                 else:
#                     params.requires_grad = True

#     def get_ignored_modules(self, mode='none'):
#         ignored_modules = []
#         if mode == 'freeze_lang':
#             for l in range(len(self.text_decoder.roberta.encoder.layer)):
#                 ignored_modules += [
#                     self.text_decoder.roberta.encoder.layer[l][0].attention,
#                     self.text_decoder.roberta.encoder.layer[l][0].intermediate,
#                     self.text_decoder.roberta.encoder.layer[l][0].output,
#                 ]
#         elif mode == 'freeze_vision':
#             for l in range(len(self.expert_encoder.transformer.resblocks)):
#                 ignored_modules += [
#                     self.expert_encoder.transformer.resblocks[l][0].attn,
#                     self.expert_encoder.transformer.resblocks[l][0].mlp,
#                     self.expert_encoder.transformer.resblocks[l][0].ln_1,
#                     self.expert_encoder.transformer.resblocks[l][0].ln_2,
#                 ]
#         elif mode == 'freeze_lang_vision':
#             for l in range(len(self.text_decoder.roberta.encoder.layer)):
#                 ignored_modules += [
#                     self.text_decoder.roberta.encoder.layer[l][0].attention,
#                     self.text_decoder.roberta.encoder.layer[l][0].intermediate,
#                     self.text_decoder.roberta.encoder.layer[l][0].output,
#                 ]
#             for l in range(len(self.expert_encoder.transformer.resblocks)):
#                 ignored_modules += [
#                     self.expert_encoder.transformer.resblocks[l][0].attn,
#                     self.expert_encoder.transformer.resblocks[l][0].mlp,
#                     self.expert_encoder.transformer.resblocks[l][0].ln_1,
#                     self.expert_encoder.transformer.resblocks[l][0].ln_2,
#                 ]
#         else:
#             ignored_modules = None
#         return ignored_modules
    
#     def forward(self, experts, caption=None, train=True, prefix=''):
#         device = experts['rgb'].device
#         if train:
#             experts_train = self.expert_encoder(experts)
#             experts_train = rearrange(experts_train, 'l b d -> b l d')  # batch_size, num_latents, output_dim

#             caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=30, return_tensors="pt").to(device)
#             answer_targets = caption.input_ids.masked_fill(caption.input_ids == self.tokenizer.pad_token_id, -100)
#             print(f"====Test experts_train : {experts_train}]")

#             # decoder

# Base model
class Prismer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = {'rgb': 3}
        for exp in config['experts']:
            if exp in ['depth', 'edge']:
                self.experts[exp] = 1
            elif exp in ['normal']:
                self.experts[exp] = 3
            elif 'seg' in exp:
                self.experts['seg'] = 64
            elif exp in ['obj_detection', 'ocr_detection']:
                self.experts[exp] = 64

        prismer_config = json.load(open('configs/prismer.json', 'r'))[config['prismer_model']]
        roberta_config = RobertaConfig.from_dict(prismer_config['roberta_model'])

        self.tokenizer = RobertaTokenizer.from_pretrained(prismer_config['roberta_model']['model_name'])
        self.expert_encoder = load_encoder(prismer_config['vit_model'], experts=self.experts, image_resolution=config['image_resolution'])
        self.text_decoder = load_decoder(prismer_config['roberta_model']['model_name'], config=roberta_config)

        self.prepare_to_train(config['freeze'])
        self.ignored_modules = self.get_ignored_modules(config['freeze'])
    
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

# With text model
class Fu_Decoder(Prismer):
    def forward()