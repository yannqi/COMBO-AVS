# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger


from detectron2.evaluation import (
    DatasetEvaluators,
    DatasetEvaluator,
    # inference_on_dataset,
    print_csv_format,
    verify_results,
)

# MaskFormer
from models import (
    BestCheckpointer,
    SemSegEvaluator,
    SemSegEvaluator_SS,
    AVSS4_SemanticDatasetMapper,
    AVSMS3_SemanticDatasetMapper,
    AVSS_SemanticDatasetMapper,
    add_maskformer2_config,
    add_audio_config,
    add_fuse_config,
    inference_on_dataset,
    inference_on_dataset_ss,
)

# from torchsummary import summary
# from thop import profile
class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:

            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        elif evaluator_type in ["sem_seg_ss"]:
            evaluator_list.append(
                SemSegEvaluator_SS(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)



    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "avss4_semantic":
            mapper = AVSS4_SemanticDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)   
        elif cfg.INPUT.DATASET_MAPPER_NAME == "avsms3_semantic":
            mapper = AVSMS3_SemanticDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)   
        elif cfg.INPUT.DATASET_MAPPER_NAME == "avss_semantic":
            mapper = AVSS_SemanticDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)       


   
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

     
        







        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if dataset_name[:5] == 'avss_':
                results_i = inference_on_dataset_ss(model, data_loader, evaluator, cfg.OUTPUT_DIR)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator, cfg.OUTPUT_DIR)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_audio_config(cfg)
    add_fuse_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR + '/vis/', distributed_rank=comm.get_rank(), name="COMBO")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        best_ckpt_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth") 
        print("Best checkpoint path: {}".format(best_ckpt_path))
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            path=best_ckpt_path, resume=False
        )      
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
