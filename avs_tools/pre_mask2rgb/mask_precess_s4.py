"""class-agnostic masks -> Maskige """
import json
import os 
from detectron2.utils.file_io import PathManager
from pycocotools import mask as mask_utils
import numpy as np
from tqdm import tqdm
from PIL import Image

import argparse


def ade_palette():
    """ADE20K palette for external use."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--split",type=str, default='train', help="train, val, test")
    parser.add_argument("--gt_dir",type=str, default='AVS_dataset/pre_SemanticSAM_mask', help="gt_dir")
    parser.add_argument("--save_dir",type=str, default='AVS_dataset/pre_SemanticSAM_mask_no_overlap', help="save_dir")
    args = parser.parse_args()
    
    # gt_dir = 'AVS_dataset/pre_SemanticSAM_mask'
    # gt_dir = 'AVS_dataset/preSAM_mask'
    # gt_dir = 'AVS_dataset/pre_mobileSAM_mask'
    
    # save_dir = 'AVS_dataset/pre_SemanticSAM_mask_no_overlap'
    # save_dir = 'AVS_dataset/preSAM_mask_no_overlap'
    # save_dir = 'AVS_dataset/pre_mobileSAM_mask_no_overlap'
    
    split = args.split
    
    gt_dir = args.gt_dir + '/AVSBench_object/Single-source/s4_data/visual_frames/' + split
    save_dir = args.save_dir + '/AVSBench_object/Single-source/s4_data/pre_SAM_mask/' + split
    categories = PathManager.ls(gt_dir)
    categories = sorted(categories)
    print(f"{len(categories)} categories found in '{gt_dir}'.")
    only_first_mask = False
    max_idx = 0


    for category in tqdm(categories):
        category_gt_dir = os.path.join(gt_dir, category)
        save_category_gt_dir = os.path.join(save_dir, category)
        videos = PathManager.ls(category_gt_dir)
        videos = sorted(videos)
        print(f"{len(videos)} videos found in '{category}'.")
        
        for video in tqdm(videos):
            video_gt_dir = os.path.join(category_gt_dir, video)
            save_video_gt_dir = os.path.join(save_category_gt_dir, video)
            basenames = PathManager.ls(video_gt_dir)
            
            image_files = []
            label_files = []
            basenames = sorted(basenames)
            
            for num_img, basename in enumerate(basenames):
                
                gt_file = os.path.join(video_gt_dir, basename)
                save_gt_file = os.path.join(save_video_gt_dir, basename)
                if basename.endswith(".png"):
                    continue
                # suffix = ".json"   # SAM
                suffix = ".npy"
                assert basename.endswith(suffix), basename  #* assert that the file is a png file
                basename = basename[: -len(suffix)]
                pre_mask = np.load(gt_file, allow_pickle=True)
                try :
                    pre_mask[0].dtype 
                except:
                    pre_int_mask = np.zeros((1, 640, 640))
                else:
                    pre_int_mask = pre_mask.astype(np.uint8)
    
                sums = np.sum(pre_int_mask, axis=(1,2))
                sorted_indices = np.argsort(sums)
                sorted_pre_mask = pre_int_mask[sorted_indices]
                # pre_mask = json.load(open(gt_file, 'r'))  #SAM
                # sorted_anns = sorted(pre_mask, key=(lambda x: x['area']), reverse=True)  #* sort the mask by area #SAM
                for idx in range(sorted_pre_mask.shape[0]):
                    
                    if idx == 0:
                        m = sorted_pre_mask[idx]
                    else:
                        m = np.where(m==0, sorted_pre_mask[idx] * (idx+1), m)
                m = m.astype(np.uint8)        
                max_id = np.max(m)
                if max_idx < max_id:
                    max_idx = max_id
                    if max_idx > 255:
                        raise ValueError('error 255')

                #save mask
                os.makedirs(save_video_gt_dir, exist_ok=True)
                save_base = os.path.join(save_video_gt_dir, basename + '_color.png')
                mask = Image.fromarray(m)
                # np.save(save_base, m)
                mask = mask.resize((224, 224), Image.NEAREST)
                color_map = ade_palette()
                color_map = np.array(color_map).astype(np.uint8)    
                mask.putpalette(color_map)
                mask.save(save_base)