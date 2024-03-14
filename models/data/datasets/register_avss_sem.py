# link: https://github.com/facebookresearch/detectron2/blob/80307d2d5e06f06a8a677cc2653f23a4c56402ac/detectron2/data/datasets/cityscapes.py

import json
import logging
import os
import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)


def _get_avss_files(df_split, root, split, pre_mask_dir=None):
    files = []
    # df_split = df_split[:100]  #! for debug
    for index in np.arange(0, len(df_split)):
        df_one_video = df_split.iloc[index]
        video_name, set = df_one_video["uid"], df_one_video["label"]

        img_base_path = os.path.join(root, set, video_name, "processed_frames")  # frames -> processed_frames
        pre_mask_path = os.path.join(pre_mask_dir, set, video_name, "processed_frames") if pre_mask_dir else None  # frames -> processed_frames
        # color_mask_base_path = os.path.join( root, set, video_name, 'labels_rgb')
        mask_base_path = os.path.join(root, set, video_name, "processed_labels_semantic")  # labels_semantic  -> processed_labels_semantic
        audio_file = os.path.join(root, set, video_name, "audio.pkl")

        # data from AVSBench-object single-source subset (5s, gt is only the first annotated frame)
        if set == "v1s":
            vid_temporal_mask_flag = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            if split == "train":
                gt_temporal_mask_flag = [1, 0, 0, 0, 0]  # , 0, 0, 0, 0, 0]
            else:
                gt_temporal_mask_flag = [1, 1, 1, 1, 1]  # , 0, 0, 0, 0, 0]
        # data from AVSBench-object multi-sources subset (5s, all 5 extracted frames are annotated)
        elif set == "v1m":
            vid_temporal_mask_flag = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            gt_temporal_mask_flag = [1, 1, 1, 1, 1]  # , 0, 0, 0, 0, 0]
        # data from newly collected videos in AVSBench-semantic (10s, all 10 extracted frames are annotated))
        elif set == "v2":
            vid_temporal_mask_flag = [1] * 10
            gt_temporal_mask_flag = [1] * 10

        img_path_list = sorted(PathManager.ls(img_base_path))
        mask_path_list = sorted(PathManager.ls(mask_base_path))
        for mask_path in mask_path_list:
            if not mask_path.endswith(".png"):
                mask_path_list.remove(mask_path)
        mask_num = len(mask_path_list)
        if split != "train":
            if set == "v2":
                assert mask_num == 10
            else:
                assert mask_num == 5

        image_files = []
        label_files = []
        pre_mask_files = []
        for img_name in img_path_list:
            image_file = os.path.join(img_base_path, img_name)
            suffix = ".jpg"
            assert img_name.endswith(suffix), img_name  # * assert that the file is a png file
            image_files.append(image_file)
            pre_mask_file = os.path.join(pre_mask_path, img_name.replace(".jpg", "_mask_color.png")) if pre_mask_path else None
            pre_mask_files.append(pre_mask_file)

        for mask_name in mask_path_list:
            label_file = os.path.join(mask_base_path, mask_name)
            label_files.append(label_file)
        files.append((image_files, label_files, audio_file, pre_mask_files, vid_temporal_mask_flag, gt_temporal_mask_flag))

    assert len(files), "No images found in {}".format(root)
    for f in files[0][:2]:
        assert PathManager.isfile(f[0] if isinstance(f, list) else f), f
    return files


def load_avss_semantic(df_split, root, split, pre_mask_dir):
    """
    Args:
        df_split: a dataframe of the split
        root (str): path to the dataset directory
        split (str): name of the split, e.g., train, val, test
        pre_mask_dir (str): path to the pre-computed mask directory
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []

    files = _get_avss_files(df_split, root, split, pre_mask_dir)

    for image_file, label_file, audio_file, pre_mask_file, vid_temporal_mask_flag, gt_temporal_mask_flag in files:
        if pre_mask_file[0] is None:
            ret.append(
                {
                    "file_names": image_file,
                    "sem_seg_file_names": label_file,
                    "audio_file_name": audio_file,
                    "vid_temporal_mask_flag": vid_temporal_mask_flag,
                    "gt_temporal_mask_flag": gt_temporal_mask_flag,
                }
            )
        else:
            ret.append(
                {
                    "file_names": image_file,
                    "sem_seg_file_names": label_file,
                    "audio_file_name": audio_file,
                    "pre_mask_file_names": pre_mask_file,
                    "vid_temporal_mask_flag": vid_temporal_mask_flag,
                    "gt_temporal_mask_flag": gt_temporal_mask_flag,
                }
            )
    assert PathManager.isfile(ret[0]["sem_seg_file_names"][0]), ret[0]["sem_seg_file_names"][0]
    return ret


def register_avss_semantic(root):
    for name, dirname in [("train", "train"), ("val", "val"), ("test", "test")]:
        df_all = pd.read_csv(os.path.join(root, "metadata.csv"), sep=",")
        df_split = df_all[df_all["split"] == dirname]
        pre_mask_dir = os.path.join(root, "pre_SAM_mask/AVSBench_semantic")
        with open(os.path.join(root, "label2idx.json"), "r") as fr:
            classes = json.load(fr)
        labels = [label for label in classes.keys()]
        name = f"avss_sem_seg_{name}"
        DatasetCatalog.register(
            name,
            lambda x=df_split, y=root, pre_mask_dir=pre_mask_dir, split=dirname: load_avss_semantic(
                df_split=x, root=y, pre_mask_dir=pre_mask_dir, split=split
            ),
        )
        MetadataCatalog.get(name).set(
            stuff_classes=labels,
            evaluator_type="sem_seg_ss",
            ignore_label=255,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
if _root.endswith("/AVSBench_semantic/"):
    logger.info(f"dataset root is '{_root}'.")
    register_avss_semantic(_root)
