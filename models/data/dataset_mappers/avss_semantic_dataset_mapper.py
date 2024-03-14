# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import pickle

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from PIL import Image
from torch.nn import functional as F


def get_v2_pallete(num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while lab > 0:
                pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
                pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
                pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    return v2_pallete


class AVSS_SemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by COMBO for Audio-Visual Segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        pre_sam,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
            pre_sam: whether to use pre-sam
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.v2_pallete = get_v2_pallete(num_cls=71)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        self.pre_sam = pre_sam

    def load_audio_lm(self, audio_lm_path):
        """Load audio log mel spectrogram from pickle file"""
        with open(audio_lm_path, "rb") as fr:
            audio_log_mel = pickle.load(fr)
        audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
        return audio_log_mel

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            if cfg.INPUT.AUGMENTATION == True:
                augs = []
                if cfg.INPUT.COLOR_AUG_SSD:
                    augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
                augs.append(T.RandomFlip())
            else:
                augs = []
        else:
            augs = []

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        # Whether to use pre sam
        pre_sam = cfg.MODEL.PRE_SAM.USE_PRE_SAM
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "pre_sam": pre_sam,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        images = []
        for image_path in dataset_dict["file_names"]:
            image = utils.read_image(image_path, format=self.img_format)
            images.append(image)

        utils.check_image_size(dataset_dict, images[0])
        if "sem_seg_file_names" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_file_name_list = dataset_dict.pop("sem_seg_file_names")

            mask_num = len(sem_seg_file_name_list)

            sem_seg_gts = []
            for sem_seg_file_name in sem_seg_file_name_list:
                sem_seg_gt = utils.read_image(sem_seg_file_name)
                sem_seg_gts.append(sem_seg_gt)

        else:
            sem_seg_gts = None
            raise ValueError("Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(dataset_dict["file_names"]))

        if ("pre_mask_file_names" in dataset_dict) and self.pre_sam:
            pre_mask_file_name_list = dataset_dict.pop("pre_mask_file_names")
            pre_mask_gts = []
            for pre_mask_file_name in pre_mask_file_name_list:
                pre_mask_gt = utils.read_image(pre_mask_file_name, format=self.img_format)
                pre_mask_gts.append(pre_mask_gt)
        else:
            pre_mask_gts = None

        only_first_mask = None
        only_first_mask = True if sum(dataset_dict["gt_temporal_mask_flag"]) == 1 else False  # * For v1s
        for num_img, image in enumerate(images):
            pre_mask_gt = pre_mask_gts[num_img] if self.pre_sam else None
            if num_img == 0:
                # * first image with random augmentation
                aug_input = T.AugInput(image, sem_seg=sem_seg_gts[num_img])
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image
                sem_seg_gt = aug_input.sem_seg
            else:
                # * other images with the same augmentation
                image = transforms.apply_image(image)
                if only_first_mask == False:
                    sem_seg_gt = transforms.apply_segmentation(sem_seg_gts[num_img])

            if self.pre_sam:
                pre_mask_gt = transforms.apply_image(pre_mask_gt)
                pre_mask_gt = torch.as_tensor(np.ascontiguousarray(pre_mask_gt.transpose(2, 0, 1)))

            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if (num_img == 0) or (only_first_mask == False):
                sem_seg_gt = torch.as_tensor(sem_seg_gt.copy())

            images[num_img] = image
            if (num_img == 0) or (only_first_mask == False):
                sem_seg_gts[num_img] = sem_seg_gt
            if self.pre_sam:
                pre_mask_gts[num_img] = pre_mask_gt

        image_shape = (images[0].shape[-2], images[0].shape[-1])  # h, w

        imgs_tensor = torch.stack(images, dim=0)  # * [10,3,224,224]
        masks_tensor = torch.stack(sem_seg_gts, dim=0).unsqueeze(dim=1)  # * [10,1,224,224]
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        dataset_dict["images"] = imgs_tensor
        dataset_dict["sem_segs"] = masks_tensor.float()

        if self.pre_sam:
            pre_masks_tensor = torch.stack(pre_mask_gts, dim=0)  # * [10,3,224,224]
            dataset_dict["pre_masks"] = pre_masks_tensor

        # Prepare per-category binary masks
        instances_list = []
        for sem_seg_gt in sem_seg_gts:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)

            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor
            instances_list.append(instances)
        dataset_dict["instances"] = instances_list
        dataset_dict["vid_temporal_mask_flag"] = torch.Tensor(dataset_dict["vid_temporal_mask_flag"])
        dataset_dict["gt_temporal_mask_flag"] = torch.Tensor(dataset_dict["gt_temporal_mask_flag"])
        # Prepare audio input
        if "audio_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            audio_file_name = dataset_dict.pop("audio_file_name")
            audio_log_mel = self.load_audio_lm(audio_file_name)
        else:
            raise ValueError("Cannot find 'audio_file_name' for semantic segmentation dataset {}.".format(dataset_dict["file_names"]))
        dataset_dict["audio_log_mel"] = audio_log_mel
        return dataset_dict
