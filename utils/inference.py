# -*- coding: utf-8 -*-
#
# This script is based on:
#
# Lixiong Qin, Mei Wang, Chao Deng, Ke Wang, Xi Chen, Jiani Hu, Weihong Deng:
# SwinFace: A Multi-task Transformer for Face Recognition, Facial Expression Recognition, Age Estimation and Face Attribute Estimation,
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 2308.11509v1, Aug. 2023.
#          see: https://arxiv.org/pdf/2308.11509.pdf
#
# The code (in the source above) is based on the SwinFace implementation of the FAC model and only serves as a part of the whole solution
# (it won't work as a standalone, it has to be replaced directly in the locally cloned SwinFace repository)
# SwinFace: https://github.com/lxq1000/SwinFace
#          license: MIT License
#


import argparse

import cv2
import numpy as np
import torch
import pandas as pd
import os
import zipfile

from model import build_model

@torch.no_grad()
def inference(cfg, weight, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    model = build_model(cfg)
    dict_checkpoint = torch.load(weight, map_location=torch.device('cpu'))
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])

    model.eval()
    output = model(img)
    return output

class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512

def extract_zip(zip_file):
    img_dir = 'img_align_celeba'

    if not os.path.exists(img_dir):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"Extracted {zip_file}")
    else:
        print(f"{img_dir} already exists. Skipping extraction.")

    return img_dir

def process_dataset(cfg, weight, img_dir, output_file):
    results = []

    # List of attributes to exclude from printing
    exclude_attributes = ["Age", "Expression", "Recognition"]

    for img_name in os.listdir(img_dir):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(img_dir, img_name)

            output = inference(cfg, weight, img_path)
            result = {"image": img_name}

            for each in output.keys():
                if each not in exclude_attributes:
                    values = output[each][0].numpy()
                    probabilities = np.exp(values) / np.sum(np.exp(values))
                    result[each] = f'{probabilities[0]},{probabilities[1]}'

            results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    cfg = SwinFaceCfg()
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str, default='swinface.pt')
    parser.add_argument('--img', type=str, default="test.jpg")
    args = parser.parse_args()

    extract_zip('img_align_celeba.zip')
    img_dir = os.path.join('img_align_celeba', 'img_align_celeba')
    process_dataset(cfg, args.weight, 'img_align_celeba', 'result.csv')
