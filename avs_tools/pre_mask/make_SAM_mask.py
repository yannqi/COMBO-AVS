import os
import sys
import numpy as np
from semantic_sam.build_semantic_sam import prepare_image, build_semantic_sam, SemanticSamAutomaticMaskGenerator, show_anns
import argparse
import json
from typing import Any, Dict, List
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import multiprocessing as mp

def args_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--sam_type", default="sam", choices=['sam', 'semantic_sam', 'mobile_sam'], type=str, help="the Sam type")
    parser.add_argument("--data_name", default="s4", type=str, help="the S4 setting")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val', 'test'],help="train or val split.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument(
    "--vis_mask",
    action="store_true",
    help=("Visualize the color mask."
    ),
    )
    parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
    )
    args = parser.parse_args()
    return args



def plot_results(outputs, image_ori, save_path):
    """
    plot input image and its reuslts
    """
    fig = plt.figure()
    plt.imshow(image_ori)
    show_anns(outputs)
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.savefig(save_path)
    plt.close()
    fig.clf()


def process_image_withPool(args, index_list, img_path, df_split, output ,vis_mask=False):
    """
    index_list: list of index of df_split
    img_path: path of images
    df_split: dataframe of split
    output: output path
    """
    if args.sam_type == "sam":
        # Load SAM   
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["default"](checkpoint="SAM_pretrained/sam_vit_h_4b8939.pth")
        sam = sam.to(device=args.device)
        sam.eval()
        mask_generator = SamAutomaticMaskGenerator(sam)  
    elif args.sam_type == "semantic_sam":
        # Load Semantic-SAM
        mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='Semantic-SAM/swinl_only_sam_many2many.pth'), level=[3]) # model_type: 'L' / 'T', depends on your checkpint
    elif args.sam_type == "mobile_sam":
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        mobile_sam = sam_model_registry["vit_t"](checkpoint="MobileSAM/weights/mobile_sam.pt")
        mobile_sam.to(device=args.device)
        mobile_sam.eval()
        mask_generator = SamAutomaticMaskGenerator(mobile_sam)  

    
    
    for index in tqdm(index_list):
        df_one_video = df_split.iloc[index]
        
        if args.data_name == "s4":
            video_name, category = df_one_video[0], df_one_video[2]   
            split = df_one_video[3]
            img_base_path =  os.path.join(img_path, split, category, video_name)
        elif args.data_name == "ms3" :
            video_name= df_one_video[0]
            img_base_path =  os.path.join(img_path, video_name)
        elif args.data_name == "avss" :
            video_name= df_one_video[1]
            split = df_one_video[5]
            label_name = df_one_video[6]
            img_base_path = os.path.join(img_path, label_name, video_name, 'processed_frames')
            
        # output 
        output_path = img_base_path.split("AVS_dataset/")[1]  
        output_path = os.path.join(output, output_path)
        os.makedirs(output_path, exist_ok=True)
        targets = [f for f in os.listdir(img_base_path) if os.path.isfile(os.path.join(img_base_path, f))]
        targets = [os.path.join(img_base_path, f) for f in targets]
        for image_pth in targets:
            base = os.path.basename(image_pth)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(output_path, base)
         
            if os.path.exists(save_base + "_mask.npy"):
                continue
            
            image = Image.open(image_pth).convert('RGB')
            image_ori = np.asarray(image)
            
            # original_image, input_image_semantic = prepare_image(image_pth=image_pth)  # change the image path to your image
            if args.sam_type == "semantic_sam" or "mobile_sam":
                input_image = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
                H, W = input_image.shape[1], input_image.shape[2]
            else: 
                input_image = torch.from_numpy(image_ori.copy()).cuda()
                H, W = input_image.shape[0], input_image.shape[1]
            masks = mask_generator.generate(input_image)
            masks2save = None
            
            for mask in masks:  
                mask = mask['segmentation']
                if masks2save is None:
                    masks2save = np.expand_dims(mask, axis=0)
                else:
                    mask = np.expand_dims(mask, axis=0)
                    masks2save = np.vstack((masks2save, mask))
            if masks2save is None:
                masks2save = np.zeros((1, H, W))
            np.save(os.path.join(save_base + "_mask.npy"), masks2save) 
    
def split_list(lst, num_processes):
    avg_len = len(lst) // num_processes
    split_lst = [lst[i*avg_len : (i+1)*avg_len] for i in range(num_processes)]
    remainder = len(lst) % num_processes
    for i in range(remainder):
        split_lst[i].append(lst[-(i+1)])
    return split_lst



if __name__ == "__main__":
    args = args_parser()
    if args.data_name == "s4":
        anno_csv = 'AVS_dataset/AVSBench_object/Single-source/s4_meta_data.csv'
        img_path = "AVS_dataset/AVSBench_object/Single-source//s4_data/visual_frames"
    elif args.data_name == "ms3" :
        anno_csv ="AVS_dataset/AVSBench_object//Multi-sources/ms3_meta_data.csv"
        img_path = "AVS_dataset/AVSBench_object//Multi-sources/ms3_data/visual_frames"
    
    elif args.data_name == "avss" :
        anno_csv = 'AVS_dataset/AVSBench_semantic/metadata.csv'
        img_path = 'AVS_dataset/AVSBench_semantic'
    split = args.split  

    df_all = pd.read_csv(anno_csv, sep=',')
    df_split = df_all[df_all['split'] == split]
    
    num_split = len(df_split)
    
    # Load Semantic-SAM 
    num_process = 4  # mutliprocess
    os.chdir(os.path.join(os.getcwd(), 'third_party/Semantic_SAM'))
    index_list = list(range(num_split))
    #random 
    # index_list = np.random.permutation(index_list)
    split_index_list = split_list(index_list, num_process)
    processes = []
    
    #! single for debug
    # process_image_withPool(args, split_index_list[0], img_path, df_split, args.output, args.vis_mask)
    #! mutliprocess
    for i in range(num_process):
        p = mp.Process(target=process_image_withPool, args=(args, split_index_list[i], img_path, df_split, args.output, args.vis_mask))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
        
        
            
            
            
            
          
             

                    



