import torch
import torch.nn.functional as F
from torch import nn
from detectron2.config import configurable
from .utils.fuse_helper import BiAttentionBlock, AttentionA2I
from ..transformer_decoder.position_encoding import PositionEmbeddingSine


# Refer from https://github.com/microsoft/GLIP/blob/main/maskrcnn_benchmark/modeling/rpn/vldyhead.py
class AVFuse(nn.Module):
    """
    Fuse the audio and visual features. 
    """

    @configurable
    def __init__(self, fused_type, audio_dim, fused_backbone, fused_backbone_dim):
        """
        Args:
        fused_type: The type of fusion. Including MHA-S, MHA-B, SCAN, FILM.
        """
        super().__init__()
        # common params
        self.fused_type = fused_type
        self.audio_dim = audio_dim
        self.fused_backbone = fused_backbone
        self.fused_backbone_dim = fused_backbone_dim
        # mha params
        self.n_head = 8
        self.embed_dim = max(self.fused_backbone_dim)

        self.hidden_dim = self.embed_dim * 4 

        self.audio_pos = nn.Embedding(1, self.audio_dim)  
        N_steps = self.fused_backbone_dim[0] // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)  
        self.level_embed = nn.Embedding(1, self.fused_backbone_dim[0])  
        print("FUSION ON, USING {}".format(self.fused_type))

        if self.fused_type == "MHA-B":
            # bi-direction (audio,image -> image,audio)
            self.b_attn = BiAttentionBlock(
                visual_features_names=self.fused_backbone,
                vision_dim_list=self.fused_backbone_dim,
                audio_dim=self.audio_dim,
                embed_dim=self.embed_dim,
                num_heads=self.n_head,
                hidden_dim=self.hidden_dim,
                dropout=0.1,
                drop_path=0.0,
                # init_values=1.0  
            )

        if self.fused_type == "MHA-S":
            self.t2i_hidden_dim = 1024  # 256 * 4
            # si-direction (audio,image -> image)
            self.a2i_attn = AttentionA2I(
                visual_features_names=self.fused_backbone,
                q_dim_list=self.fused_backbone_dim,
                k_dim=self.audio_dim,
                embed_dim=self.embed_dim,
                num_heads=self.n_head,
                hidden_dim=self.t2i_hidden_dim,
                dropout=0.1,
                drop_path=0.0,
                # init_values=1.0 / 8, 
            )

        if self.fused_type == "MHA-S-Audio":
            # bi-direction (audio,image -> image,audio)
            self.b_attn = BiAttentionBlock(
                visual_features_names=self.fused_backbone,
                vision_dim_list=self.fused_backbone_dim,
                audio_dim=self.audio_dim,
                embed_dim=self.embed_dim,
                num_heads=self.n_head,
                hidden_dim=self.hidden_dim,
                dropout=0.1,
                drop_path=0.0,
                # init_values=1.0 
            )

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "fused_type": cfg.MODEL.FUSE_CONFIG.TYPE,
            "audio_dim": cfg.MODEL.FUSE_CONFIG.AUDIO_DIM,
            "fused_backbone": cfg.MODEL.FUSE_CONFIG.FUSED_BACKBONE,
            "fused_backbone_dim": cfg.MODEL.FUSE_CONFIG.FUSED_BACKBONE_DIM,
        }
        return ret

    def __call__(self, visual_features, audio_features):
        fused_visual_features = None
        fused_audio_features = None

        # * audio position embedding
        audio_pos = self.audio_pos.weight.unsqueeze(1).repeat(1, audio_features.shape[0], 1)
        audio_pos = audio_pos.permute(1, 0, 2).contiguous()

        size_list = []
        for i in range(len(visual_features)):
            size_list.append(visual_features[self.fused_backbone[i]].shape[-2:])
            image_pos = (self.pe_layer(visual_features[self.fused_backbone[i]], None)).flatten(2).permute(0, 2, 1).contiguous()
            visual_features[self.fused_backbone[i]] = (
                (visual_features[self.fused_backbone[i]]).flatten(2) + self.level_embed.weight[i][None, :, None]
            ).reshape(visual_features[self.fused_backbone[i]].shape)

        if self.fused_type == "MHA-B":
            fused_visual_features, fused_audio_features = self.b_attn(visual_features, audio_features, pos_v=image_pos, pos_a=audio_pos)
        elif self.fused_type == "MHA-S":
            # audio, image -> image
            fused_visual_features = self.a2i_attn(
                q_list=visual_features, k=audio_features, v=audio_features, attention_mask=None  # ? Attention mask is none now
            )
            fused_audio_features = audio_features
        elif self.fused_type == "MHA-S-Audio":
            # audio, image -> audio
            fused_visual_features, fused_audio_features = self.b_attn(visual_features, audio_features, pos_v=image_pos, pos_a=audio_pos)
            fused_visual_features = visual_features
        elif self.fused_type == "MHA-None":
            fused_visual_features = visual_features
            fused_audio_features = audio_features

        features_dict = {"visual": fused_visual_features, "audio": fused_audio_features}
        return features_dict
