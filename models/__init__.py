# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_audio_config, add_fuse_config

# dataset loading


from .data.dataset_mappers.avss4_semantic_dataset_mapper import (
    AVSS4_SemanticDatasetMapper,
)
from .data.dataset_mappers.avsms3_semantic_dataset_mapper import (
    AVSMS3_SemanticDatasetMapper,
)
from .data.dataset_mappers.avss_semantic_dataset_mapper import (
    AVSS_SemanticDatasetMapper,
)


# models
from .maskformer_model import MaskFormer

# evaluation
from .evaluation.sem_seg_evaluation import SemSegEvaluator
from .evaluation.sem_seg_evaluation_ss import SemSegEvaluator_SS
from .evaluation.evaluator import inference_on_dataset, inference_on_dataset_ss

# hook
from .engine.hooks import BestCheckpointer
