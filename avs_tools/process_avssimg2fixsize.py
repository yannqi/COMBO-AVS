import json
import os 
from detectron2.utils.file_io import PathManager
from pycocotools import mask as mask_utils
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp

def crop_resize_img(crop_size, img, is_mask=False):
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    # print("crop for train. set")
    return img

def resize_img(crop_size, img, is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

    
def split_list(lst, num_processes):
    avg_len = len(lst) // num_processes
    split_lst = [lst[i*avg_len : (i+1)*avg_len] for i in range(num_processes)]
    remainder = len(lst) % num_processes
    for i in range(remainder):
        split_lst[i].append(lst[-(i+1)])
    return split_lst
    

    
def precess_ss(index_list, img_path, pre_sam_path, df_split):
    
    for index in tqdm(index_list):
        df_one_video = df_split.iloc[index]
        video_name= df_one_video[1]
        split = df_one_video[5]
        label_name = df_one_video[6]
        img_base_path = os.path.join(img_path, label_name, video_name, 'frames')
        mask_base_path = os.path.join(img_path, label_name, video_name, 'labels_semantic')
        sam_base_path = os.path.join(pre_sam_path, label_name, video_name, 'frames')
        saved_img_base_path = os.path.join(img_path, label_name, video_name, 'processed_frames')
        saved_mask_base_path = os.path.join(img_path, label_name, video_name, 'processed_labels_semantic')
        saved_pre_sam_base_path = os.path.join(pre_sam_path, label_name, video_name, 'processed_frames')
        basenames = PathManager.ls(img_base_path)
        pre_sam_basenames = PathManager.ls(sam_base_path)
        mask_basenames = PathManager.ls(mask_base_path)
        basenames = sorted(basenames)
        mask_basenames = sorted(mask_basenames)
        pre_sam_basenames = sorted(pre_sam_basenames)
        os.makedirs(saved_img_base_path, exist_ok=True)
        os.makedirs(saved_mask_base_path, exist_ok=True)
        os.makedirs(saved_pre_sam_base_path, exist_ok=True)
        for num_img, basename in enumerate(basenames):
            
            img_file = os.path.join(img_base_path, basename)
            save_img_file = os.path.join(saved_img_base_path, basename)
            
            pre_sam_file = os.path.join(sam_base_path, pre_sam_basenames[num_img])
            save_pre_sam_file = os.path.join(saved_pre_sam_base_path, pre_sam_basenames[num_img])
            
            ori_img = Image.open(img_file).convert('RGB')
            ori_pre_sam = Image.open(pre_sam_file).convert('RGB')
            
            if split == 'train':
                img = crop_resize_img(224, ori_img, is_mask=False)
                pre_sam = crop_resize_img(224, ori_pre_sam, is_mask=True)

            else:
                img = resize_img(224, ori_img, is_mask=False)
                pre_sam = resize_img(224, ori_pre_sam, is_mask=True)
               
        
            #save 
            img.save(save_img_file)
            pre_sam.save(save_pre_sam_file)

        for num_mask, mask_basename in enumerate(mask_basenames):
            mask_file = os.path.join(mask_base_path, mask_basename)
            save_mask_file = os.path.join(saved_mask_base_path, mask_basename)
            ori_mask = Image.open(mask_file)
            if split == 'train':
                mask = crop_resize_img(224, ori_mask, is_mask=True)
            else:
                mask = resize_img(224, ori_mask, is_mask=True)
            mask.save(save_mask_file)
            
            
if __name__ == "__main__":
            
        

    


    anno_csv = 'AVS_dataset/AVSBench_semantic/metadata.csv'
    img_path = 'AVS_dataset/AVSBench_semantic'
    pre_sam_path = 'AVS_dataset/AVSBench_semantic/pre_SAM_mask/AVSBench_semantic'
    df_all = pd.read_csv(anno_csv, sep=',')
    df_split = df_all[df_all['split'] == 'test']
    num_split = len(df_split)

    num_process = 10  
    index_list = list(range(num_split))
    split_index_list = split_list(index_list, num_process)

    processes = []

    for i in range(num_process):
        p = mp.Process(target=precess_ss, args=(split_index_list[i], img_path, pre_sam_path, df_split))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
     
     