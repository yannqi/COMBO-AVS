# Copyright (c) Facebook, Inc. and its affiliates.
# link: https://github.com/facebookresearch/detectron2/blob/80307d2d5e06f06a8a677cc2653f23a4c56402ac/detectron2/data/datasets/cityscapes.py
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import torch
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image
import pandas as pd
import pickle
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from tqdm import tqdm
import sys; sys.path.append(os.getcwd())
from models.modeling.audio_backbone.torchvggish import vggish_input

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)
def _get_avss_files(df_split, root, split):
    files = []
    for index in tqdm(np.arange(0,len(df_split))):
        df_one_video = df_split.iloc[index]
        video_name, set = df_one_video['uid'], df_one_video['label']
       
        audio_path = os.path.join( root, set, video_name, 'audio.wav')
   
        # data from AVSBench-object single-source subset (5s, gt is only the first annotated frame)
        if set == 'v1s':
            vid_temporal_mask_flag = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # .bool()
            gt_temporal_mask_flag = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # .bool()
        # data from AVSBench-object multi-sources subset (5s, all 5 extracted frames are annotated)
        elif set == 'v1m':
            vid_temporal_mask_flag = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # .bool()
            gt_temporal_mask_flag = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # .bool()
        # data from newly collected videos in AVSBench-semantic (10s, all 10 extracted frames are annotated))
        elif set == 'v2':
            vid_temporal_mask_flag = [1] * 10  # .bool()
            gt_temporal_mask_flag = [1] * 10  # .bool()
        # audio_path = 'AVS_dataset/AVSBench_semantic/v2/-HeRZSisQLY_10000_20000/audio.wav'
        x = vggish_input.wavfile_to_examples(audio_path)
        new_x = None

        if x.shape[0] != 10:
            new_x = torch.zeros((10,1 ,96, 64))
            new_x[:x.shape[0]] = x
        else:
            new_x = x
        with open(audio_path.replace('.wav', '.pkl'), "wb") as fw:
            pickle.dump(new_x, fw)
        files.append((audio_path, vid_temporal_mask_flag, gt_temporal_mask_flag))
    
    return files

def load_avss_semantic(df_split, root, split):
    """
        Args:
            img_dir (str): path to the image directory.
            mask_dir (str): path to the mask directory.
        Returns:
            list[dict]: a list of dicts in Detectron2 standard format. each has "file_name" and
                "sem_seg_file_name".
    """
    ret = []

    files = _get_avss_files(df_split, root, split)
      
    print('len(files): ', len(files))


if __name__=="__main__":

    img_dir = 'AVS_dataset/AVSBench_semantic'
    splits = ["train","val","test"]
    df_all = pd.read_csv( os.path.join(img_dir,'metadata.csv'), sep=',')
   
    for split in splits:
        df_split = df_all[df_all['split'] == split]
        ret = load_avss_semantic(df_split=df_split, root=img_dir, split=split)
    