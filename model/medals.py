import random
from typing import Tuple, Union, List

import torch.nn as nn
import torch.nn.functional as F
import torch 
from einops import rearrange, repeat, reduce
from positional_encodings.torch_encodings import PositionalEncoding3D
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from .transformer_decoder import TransformerDecoder,TransformerDecoderLayer

class MedalS(nn.Module):
    def __init__(self, vision_backbone='UNET', input_channels=1, image_size=[224, 224, 128], patch_size=[32, 32, 32], deep_supervision=False):
        """
        Args:
            vision_backbone (str, optional): visual backbone. Defaults to UNET.
            image_size (list, optional): image size. Defaults to [224, 224, 128].
            patch_size (list, optional): maxium downsample ratio of the bottleneck feature map. Defaults to [32, 32, 32].
            deep_supervision (bool, optional): seg results from mid layers of decoder. Defaults to False.
        """
        super().__init__()
        image_height, image_width, frames = image_size
        self.hw_patch_size = patch_size[0] 
        self.frame_patch_size = patch_size[-1]
        
        self.deep_supervision = deep_supervision
        
        # backbone can be any multi-scale enc-dec vision backbone
        # the enc outputs multi-scale latent features
        # the dec outputs multi-scale per-pixel features
        self.backbone = {
            'UNET' : PlainConvUNet(input_channels=input_channels, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=deep_supervision,
                                   nonlin_first=False
                                   )
        }[vision_backbone]
        
        self.backbone.apply(InitWeights_He(1e-2))
        
        # fixed to text encoder out dim
        query_dim = 768

        # all backbones are 6-depth, thus the first 5 scale latent feature outputs need to be down-sampled
        self.avg_pool_ls = [    
            nn.AvgPool3d(32, 32),
            nn.AvgPool3d(16, 16),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(4, 4),
            nn.AvgPool3d(2, 2),
            ]
            
        # multi-scale latent feature are projected to query_dim before query decoder
        self.projection_layer = {
            'UNET' : nn.Sequential(
                        nn.Linear(1792, 768),
                        nn.GELU(),
                        nn.Linear(768, query_dim),
                        nn.GELU()
                    ),
        }[vision_backbone]
        
        # positional encoding
        pos_embedding = PositionalEncoding3D(query_dim)(torch.zeros(1, (image_height//self.hw_patch_size), (image_width//self.hw_patch_size), (frames//self.frame_patch_size), query_dim)) # b h/p w/p d/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w d c -> (h w d) b c')   # n b dim
        
        # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=query_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(query_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        
        self.query_proj = nn.Identity()
        
        # mask embedding are projected to perpixel_dim
        # mid stage output (only consider the last 3 mid layers of decoder, i.e. feature maps with resolution /2 /4 /8)
        if self.deep_supervision:
            feature_per_stage = {
                'UNET':[64, 128, 256]
                }[vision_backbone]
            mid_dim = {
                'UNET':[256, 384, 512]
                }[vision_backbone]
            self.mid_mask_embed_proj = []
            for hidden_dim, per_pixel_dim in zip(mid_dim, feature_per_stage):
                self.mid_mask_embed_proj.append(
                    nn.Sequential(
                        nn.Linear(query_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, per_pixel_dim),
                        nn.GELU(),
                        ),
                    )
                self.mid_mask_embed_proj = nn.ModuleList(self.mid_mask_embed_proj)
                
        # largest output        
        mid_dim, per_pixel_dim = {
            'UNET' : [256, 64]
        }[vision_backbone]
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(query_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, per_pixel_dim),
            nn.GELU(),
            )

        self.fusion_conv = StackedConvBlocks(
                1, nn.Conv3d, 2 * per_pixel_dim, per_pixel_dim,
                3, 1, True, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True},
                None, None, nn.LeakyReLU, None, False)

    def enhance_with_coarse_pred(self, pixel_emb, mask_emb, coarse_pred):
        """
        Enhance pixel embeddings with coarse prediction information
        
        Args:
            pixel_emb (torch.tensor): B,C,H,W,D per-pixel embeddings
            mask_emb (torch.tensor): B,N,C mask embeddings
            coarse_pred (torch.tensor): B,N,H,W,D coarse prediction probabilities
            
        Returns:
            torch.tensor: enhanced pixel embeddings B,C,H,W,D
        """
        
        # [B,N,C] -> [B,C,N]
        mask_emb_transposed = mask_emb.permute(0, 2, 1)  
        
        enhanced_emb = torch.einsum('bnhwd,bcn->bchwd', coarse_pred, mask_emb_transposed)
        
        combined = torch.cat([pixel_emb, enhanced_emb], dim=1)

        enhanced_pixel_emb = self.fusion_conv(combined)  # B,C,H,W,D
        
        return enhanced_pixel_emb
            
    def vision_backbone_forward(self, image_input):
        """
        Visual backbone forward

        Args:
            image_input (torch.tensor): C,H,W,D (C=1)

        Returns:
            image_embedding (torch.tensor): multiscale image features from encoder layers. N,B,d
            pos (torch.tensor): position encoding. N,B,d
            per_pixel_embedding_ls (List of torch.tensor): perpixel embeddings from decoder layers. B,d,H,W,D
        """

        # Image Encoder and Pixel Decoder
        latent_embedding_ls, per_pixel_embedding_ls = self.backbone(image_input) # B Dim H/P W/P D/P
        
        # avg pooling each multiscale feature to H/P W/P D/P
        image_embedding = []
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
            tmp = avg_pool(latent_embedding)
            image_embedding.append(tmp)   # B ? H/P W/P D/P
        image_embedding.append(latent_embedding_ls[-1])

        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)
        image_embedding = rearrange(image_embedding, 'b d h w depth -> b h w depth d')
        image_embedding = self.projection_layer(image_embedding)   # B H/P W/P D/P Dim
        image_embedding = rearrange(image_embedding, 'b h w d dim -> (h w d) b dim') # (H/P W/P D/P) B Dim
            
        # add pe to image embedding
        pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P D/P) B Dim
            
        return image_embedding, pos, per_pixel_embedding_ls 

    def infer_forward(self, q, image_embedding, pos, per_pixel_embedding_ls, simulated_lowres_mc_pred=None):
        """
        infer batches of queries (a list) on a batch of patches
        
        Args:
            q (List of torch.tensor): N,d
            simulated_lowres_mc_pred (torch.tensor, optional): B,N,H,W,D low-res multi-channel prediction

        Returns:
            logits (torch.tensor): concat seg output of all queries. B,N_all,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        # query decoder
        N,_ = q.shape    # N is the num of query
        q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim NOTE:By default, attention in torch is not batch_first
        q = self.query_proj(q)
        mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        
        # Dot product
        mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
        mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
        
        # Enhance features with low-res multi-channel prediction if available
        if simulated_lowres_mc_pred is not None:
            per_pixel_embedding = self.enhance_with_coarse_pred(
                per_pixel_embedding, 
                mask_embedding,
                simulated_lowres_mc_pred)
        
        logits = torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding) # bnhwd
        
        return logits
    
    def train_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls, simulated_lowres_mc_pred=None):
        """
        Args:
            queries (torch.tensor): B,N,d
            simulated_lowres_mc_pred (torch.tensor, optional): B,N,H,W,D low-res multi-channel prediction

        Returns:
            logits (List of torch.tensor): list of seg results. B,N,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        # query decoder
        _, N, _ = queries.shape    # N is the num of query
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim NOTE:By default, attention in torch is not batch_first
        queries = self.query_proj(queries)
        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        
        # Dot product
        last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
        last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
        
        # Enhance features with low-res multi-channel prediction if available
        if simulated_lowres_mc_pred is not None:
            per_pixel_embedding = self.enhance_with_coarse_pred(
                per_pixel_embedding,
                last_mask_embedding,
                simulated_lowres_mc_pred)
        
        logits = [torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, last_mask_embedding)]
        
        # deep supervision
        if self.deep_supervision:
            for mask_embed_proj, per_pixel_embedding in zip(self.mid_mask_embed_proj, per_pixel_embedding_ls[1:]):  # H/2 --> H/16
                mid_mask_embedding = mask_embed_proj(mask_embedding)
                mid_mask_embedding = rearrange(mid_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
                
                logits.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mid_mask_embedding))
                
        return logits

    def forward(self, queries, image_input, simulated_lowres_sc_pred=None, simulated_lowres_mc_pred=None, train_mode=True):
        # Handle single-channel low-res prediction if provided
        if simulated_lowres_sc_pred is not None:
            # concatenate image and simulated low-res single channel prediction
            image_input = torch.cat([image_input, simulated_lowres_sc_pred], dim=1)  # b2whd

        # get vision features
        image_embedding, pos, per_pixel_embedding_ls = self.vision_backbone_forward(image_input)
            
        # Train Forward -----------------------------------------------------------------------
        if train_mode:
            logits = self.train_forward(queries, image_embedding, pos, per_pixel_embedding_ls, simulated_lowres_mc_pred)
            
        # Infer / Evaluate Forward ------------------------------------------------------------
        else:
            del image_input
            torch.cuda.empty_cache()
            logits = self.infer_forward(queries, image_embedding, pos, per_pixel_embedding_ls, simulated_lowres_mc_pred)
        
        return logits

if __name__ == '__main__':
    model = MedalS().cuda()    
    image = torch.rand((1, 3, 288, 288, 96)).cuda()
    query = torch.rand((2, 10, 768)).cuda()
    segmentations = model(query, image)
    print(segmentations.shape)