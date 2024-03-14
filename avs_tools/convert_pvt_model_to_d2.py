# Convert pvt model from PyTorch to Detectron2 format.
import os
import argparse

import torch
import pickle as pkl
import sys




   
def parse_args():
    parser = argparse.ArgumentParser("D2 model converter")

    parser.add_argument("--source_model", default="", type=str, help="Path or url to the model to convert")
    parser.add_argument("--output_model", default="", type=str, help="Path where to save the converted model")
    return parser.parse_args()


def main():
    args = parse_args()
    obj = torch.load(args.source_model, map_location="cpu")
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            # k = "backbone." + k
            k = k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "PVT_V2_B5", "matching_heuristics": True}

    with open(args.output_model, "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
if __name__ == "__main__":
    main()
    
    
 
   

   