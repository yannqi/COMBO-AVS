import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath
from typing import Optional
from torch import nn, Tensor


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2).contiguous()
    layer = layer.reshape(N, -1, C)
    return layer


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2).contiguous()

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2).contiguous()

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, a_dim, embed_dim, num_heads, dropout=0.1):
        """
        v_dim: visual feature dimension
        a_dim: audio feature dimension
        embed_dim: embedding dimension
        num_heads: number of heads
        """
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.a_dim = a_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.a_proj = nn.Linear(self.a_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_a_proj = nn.Linear(self.a_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_a_proj = nn.Linear(self.embed_dim, self.a_dim)

        self.stable_softmax_2d = False  
        self.clamp_min_for_underflow = True  
        self.clamp_max_for_overflow = True 

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a_proj.weight)
        self.a_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_a_proj.weight)
        self.values_a_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_a_proj.weight)
        self.out_a_proj.bias.data.fill_(0)

    def forward(self, v, a, pos_v=None, pos_a=None):
        bsz, tgt_len, embed_dim = v.size()  # * bs*5, 56*56, 256

        if pos_v is None:
            query_states = self.v_proj(v) * self.scale  # * [bs*5, 56*56, 256] -> Linear(256, 2048) -> [bs*5, 56*56, 2048]
        else:
            query_states = (self.v_proj(v + pos_v)) * self.scale

        if pos_a is None:
            key_states = self._shape(self.a_proj(a), -1, bsz)  # * [bs*5, 128] -> Linear(128, 2048) -> [bs*5, 1, 2048] -> [bs*5, 8, 1, 256]
        else:
            key_states = self._shape((self.a_proj(a + pos_a)), -1, bsz)

        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)  # * [bs*5, 56*56, 256] -> Linear(256, 2048) -> [bs*5, 8, 56*56, 256]
        value_a_states = self._shape(self.values_a_proj(a), -1, bsz)  # * [bs*5, 128] -> Linear(128, 2048) -> [bs*5, 8, 1, 256]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # * bs*5*8, -1, 256
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)  # * [bs*5*8, 56*56, 256] vision feature = query
        key_states = key_states.view(*proj_shape)  # * [bs*5*8, 1, 256]  audio feature = key
        # * for bi- attention, we need the value_v_states and value_a_states
        value_v_states = value_v_states.view(*proj_shape)  # * [bs*5*8, 56*56, 256] vision feature = value
        value_a_states = value_a_states.view(*proj_shape)  # * [bs*5*8, 1, 256]  audio feature = value

        src_len = key_states.size(1)  # * 1
        # * torch.bmm: input（p,m,n) * mat2(p,n,a) ->output(p,m,a)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2).contiguous())  # * Q x K^T [bs*5*8, 56*56, 1]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}")

        # attn_weights_a = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:  # * False
            attn_weights = attn_weights - attn_weights.max()
        # * The purpose of this operation is to prevent underflow or overflow during numerical calculations, which can lead to inaccurate results or errors.
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2).contiguous()  # * [bs*5*8, 1, 56*56]
        # Max-Normalization
        attn_weights_a = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]  # * [bs*5*8, 1, 56*56]
        if self.clamp_min_for_underflow:
            attn_weights_a = torch.clamp(attn_weights_a, min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_a = torch.clamp(attn_weights_a, max=50000)  # Do not increase 50000, data type half has quite limited range
        attn_weights_a = attn_weights_a.softmax(dim=-1)  # * Softmax(attn) [bs*5*8, 1, 56*56] 
        attn_weights_v = nn.functional.softmax(attn_weights, dim=1)  #!  Softmax(attn)  [bs*5*8, 56*56, 1] 
        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_a = F.dropout(attn_weights_a, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(
            attn_probs_v, value_a_states
        )  # * attn x V_audio -> [bs*5*8, 56*56, 1] x [bs*5*8, 1, 256] = [bs*5*8, 56*56, 256]
        attn_output_a = torch.bmm(
            attn_probs_a, value_v_states
        )  # * attn^T x V_visual -> [bs*5*8, 1, 56*56] x [bs*5*8, 56*56, 256] = [bs*5*8, 1, 256]

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_a.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_a` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_a.size()}"
            )

        # reshape back to initial dim

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).contiguous()
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)  # * [bs*5, 56*56, 2048]

        attn_output_a = attn_output_a.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_a = attn_output_a.transpose(1, 2).contiguous()
        attn_output_a = attn_output_a.reshape(bsz, src_len, self.embed_dim)  # * [bs*5, 1, 2048]

        attn_output_v = self.out_v_proj(attn_output_v)  # * Linear(2048,256)  -> [bs*5, 56*56, 256]
        attn_output_a = self.out_a_proj(attn_output_a)  # * Linear(2048,128)  -> [bs*5, 1, 128]

        return attn_output_v, attn_output_a


class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        visual_features_names,
        vision_dim_list,
        audio_dim,
        embed_dim,
        num_heads,
        hidden_dim=None,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        self.visual_features_names = visual_features_names

        # pre layer norm
        self.layer_norm_v_list = nn.ModuleList()
        self.layer_norm_a_list = nn.ModuleList()
        self.attn_list = nn.ModuleList()

        self.gamma_v_list = nn.ParameterList()
        for vision_dim in vision_dim_list:
            self.layer_norm_v_list.append(nn.LayerNorm(vision_dim))
            self.layer_norm_a_list.append(nn.LayerNorm(audio_dim))
            self.attn_list.append(
                BiMultiHeadAttention(
                    v_dim=vision_dim,
                    a_dim=audio_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            # add layer scale for training stability
            self.gamma_v_list.append(nn.Parameter(init_values * torch.ones((vision_dim)), requires_grad=True))

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_a = nn.Parameter(init_values * torch.ones((audio_dim)), requires_grad=True)

    def forward(self, visual_features, audio_feature, pos_a=None, pos_v=None):
        size_per_level = []
        new_v_list, new_a_list = [], []
        for num_level, feature_name in enumerate(self.visual_features_names):
            feat_per_level = visual_features[feature_name]
            bs, c, h, w = feat_per_level.shape  # * [bs*5, 256, 56, 56] / [bs*5, 512, 28, 28] / [bs*5, 1024, 14, 14] / [bs*5, 2048, 7, 7]
            size_per_level.append([h, w])
            visual_feature_flatten = permute_and_flatten(feat_per_level, bs, 1, c, h, w)  # * [bs*5, 56*56, 256]
            # * start fusion
            new_v, new_a = self.single_attention_call(v=visual_feature_flatten, a=audio_feature, level=num_level, pos_a=pos_a, pos_v=pos_v)
            # * [bs*5, 56*56, 256] [bs*5, 128]
            # [bs, N, C] -> [bs, C, N]
            new_v = new_v.transpose(1, 2).contiguous()
            new_v_list.append(new_v)
            new_a_list.append(new_a)
            # visual_features_flatten.append(feature_flatten)
        # visual_features_flatten = torch.cat(visual_features_flatten, dim=1)  #* [bs*5, 56*56, 256]

        # * new audio_feature
        new_a = torch.stack(new_a_list, dim=1)  # * [bs*5, 4, 128]
        audio_feature = torch.mean(new_a, dim=1)  # * [bs*5, 128]

        # * new vision_feature
       
        for num_level, (h, w) in enumerate(size_per_level):
            new_v_per_level = new_v_list[num_level].view(bs, -1, h, w).contiguous()
            visual_features[self.visual_features_names[num_level]] = new_v_per_level

        return visual_features, audio_feature

    def single_attention_call(self, v, a, level, pos_v=None, pos_a=None):
        """
        Args:
            v: visual feature
            a: audio feature
        """
        v = self.layer_norm_v_list[level](v)  # * [bs*5,56*56, 256]
        a = self.layer_norm_a_list[level](a)  # * [bs*5, 1, 128]
        delta_v, delta_a = self.attn_list[level](v, a, pos_v, pos_a)  # BiMultiHeadAttention
        # delta_a = delta_a.squeeze(1)  #* [bs*5, 1, 128] -> [bs*5, 128]    
        v = v + self.drop_path(self.gamma_v_list[level] * delta_v)
        a = a + self.drop_path(self.gamma_a * delta_a)
        return v, a # * [bs*5, 56*56, 256] [bs*5, 128]


# Single Direction MHA
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)

        self.stable_softmax_2d = False  # cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D #* False
        self.clamp_min_for_underflow = True  # cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW  #* True
        self.clamp_max_for_overflow = True  # cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW  #* True
        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2).contiguous())

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}")

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000)  # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert attention_mask.dim() == 2
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}")
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class AttentionA2I(nn.Module):
    def __init__(
        self,
        visual_features_names,
        q_dim_list,
        k_dim,
        embed_dim,
        num_heads,
        hidden_dim=None,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
    ):
        """
        Inputs:
            q_dim - Dimensionality of query feature vectors, which is the dim of Visual feature.
            k_dim - Dimensionality of key feature vectors, which is the dim of Audio feature.
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(AttentionA2I, self).__init__()

        self.visual_features_names = visual_features_names

        # pre_layer norm
        self.layer_norm_k_1_list = nn.ModuleList()
        self.layer_norm_q_1_list = nn.ModuleList()
        self.attn_list = nn.ModuleList()
        self.gamma_list = nn.ParameterList()
        for q_dim in q_dim_list:
            self.layer_norm_q_1_list.append(nn.LayerNorm(q_dim))
            self.layer_norm_k_1_list.append(nn.LayerNorm(k_dim))
            self.attn_list.append(
                MultiHeadAttention(
                    q_dim=q_dim,
                    k_dim=k_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
            )

            # add layer scale for training stability
            self.use_layer_scale = True  # * temp true now
            if self.use_layer_scale:
                self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
                self.gamma_list.append(nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True))

    def forward(self, q_list, k, v, attention_mask):
        """Args:
        q_list: list of query feature vectors, which is the dim of Visual feature.
        k: key feature vectors, which is the dim of Audio feature.
        v: value feature vectors, which is the dim of Audio feature.
        """
        qs = []
        size_per_level = []
        for q_index, name in enumerate(self.visual_features_names):
            q = q_list[name]
            bs, _, h, w = q.shape
            size_per_level.append([h, w])
            # (batch, seq_len, embed_size)
            q = q.flatten(2).transpose(1, 2)
            q = self.layer_norm_q_1_list[q_index](q)
            k, v = self.layer_norm_k_1_list[q_index](k), self.layer_norm_k_1_list[q_index](v)
            delta_q = self.attn_list[q_index](q, k, v, attention_mask=attention_mask)[0]
            if self.use_layer_scale:
                q = q + self.drop_path(self.gamma_list[q_index] * delta_q)
            else:
                q = q + delta_q
            q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)
            qs.append(q)

        # * new vision_feature
        for num_level, (h, w) in enumerate(size_per_level):
            new_v_per_level = qs[num_level].view(bs, -1, h, w).contiguous()
            q_list[self.visual_features_names[num_level]] = new_v_per_level

        return q_list
