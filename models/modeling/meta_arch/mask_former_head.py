import logging
from typing import Callable, Dict, List, Optional, Tuple, Union


from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder
from ..fusion_module.AVFuse import AVFuse
from ..misc.audio_transformation import audio_mlp


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):
    _version = 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        fusion_module: nn.Module or None,
        audio_transformation: nn.Module or None,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        # * Fusion module
        if fusion_module is not None:
            self.late_fusion = True
            self.fusion_module = fusion_module  # * add fusion module
            self.audio_transformation = audio_transformation
        else:
            self.late_fusion = False

        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        # fusion module for late fusion
        if cfg.MODEL.FUSE_CONFIG.FUSION_STEP == "late":
            if cfg.MODEL.FUSE_CONFIG.QUERIES_FUSE_TYPE == "dim":
                audio_out_dim = 128  
            else:
                audio_out_dim = 256  
            cfg.defrost()
            cfg.MODEL.FUSE_CONFIG.AUDIO_OUT_DIM = audio_out_dim  
            cfg.freeze()
            fusion_module = AVFuse(cfg)
            # audio transformation
            audio_transformation = audio_mlp(in_dim=128, middle_dim=4096, out_dim=audio_out_dim)
        else:
            fusion_module = None
            audio_transformation = None

        return {
            "input_shape": {k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES},
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "fusion_module": fusion_module,  # * add fusion module
            "audio_transformation": audio_transformation,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, audio_features, mask=None):
        return self.layers(features, audio_features, mask)

    def layers(self, features, audio_feature, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        if self.late_fusion:
            # * Late fusion
            fused_visual_features = {}
            fused_visual_features["res2"] = mask_features  # * Only fuse the mask features, for convenience, we use 'res2' as the key.
            fusion_feature = self.fusion_module(fused_visual_features, audio_feature)
            fused_visual_features = fusion_feature["visual"]
            fusion_audio_feature = fusion_feature["audio"]
            fusion_audio_feature = self.audio_transformation(fusion_audio_feature)  # * [bs*5, 256*N]
            if self.transformer_in_feature == "multi_scale_pixel_decoder":
                predictions = self.predictor(multi_scale_features, fusion_audio_feature, fused_visual_features["res2"], mask)

        else:
            if self.transformer_in_feature == "multi_scale_pixel_decoder":
                predictions = self.predictor(multi_scale_features, audio_feature, mask_features, mask)

        return predictions
