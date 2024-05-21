# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) Make-An-Audio2 Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

import pathlib
import sys
import os
directory = pathlib.Path(os.getcwd())
sys.path.append(str(directory))
import torch
import numpy as np
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
import torch.nn.functional as F
import argparse
import csv
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',type=str,default='')
    parser.add_argument('--wavsdir',type=str)
    parser.add_argument('--mean',type=bool,default=True)
    parser.add_argument('--ckpt_path', default="useful_ckpts/CLAP")
    args = parser.parse_args()
    return args

def add_audio_path(df):
    df['audio_path'] = df.apply(lambda x:x['mel_path'].replace('.npy','.wav'),axis=1)
    return df

def build_csv_from_wavs(root_dir):
    with open('ldm/data/audiocaps_fn2cap.json','r') as f:
        fn2cap = json.load(f)
    wavs_root = os.path.join(root_dir,'fake_class')
    wavfiles = os.listdir(wavs_root)
    wavfiles = list(filter(lambda x:x.endswith('.wav') and x[-6:-4]!='gt',wavfiles))
    print(len(wavfiles))
    dict_list = []
    for wavfile in wavfiles:
        tmpd = {'audio_path':os.path.join(wavs_root,wavfile)}
        key = wavfile.rsplit('_sample')[0] + wavfile.rsplit('_sample')[1][:2]
        tmpd['caption'] = fn2cap[key]
        dict_list.append(tmpd)
    df = pd.DataFrame.from_dict(dict_list)
    csv_path = f'{os.path.basename(root_dir)}.csv'
    csv_path = os.path.join(wavs_root,csv_path)
    df.to_csv(csv_path,sep='\t',index=False)
    return csv_path

def cal_score_by_csv(csv_path,clap_model):    # audiocaps val的gt音频的clap_score计算为0.479077
    df = pd.read_csv(csv_path,sep='\t')
    clap_scores = []
    if not ('audio_path' in df.columns):
        df = add_audio_path(df)
    caption_list,audio_list = [],[]
    with torch.no_grad():
        for idx,t in enumerate(tqdm(df.itertuples()),start=1): 
            # text_embeddings = clap_model.get_text_embeddings([getattr(t,'caption')])# 经过了norm的embedding
            # audio_embeddings = clap_model.get_audio_embeddings([getattr(t,'audio_path')], resample=True)
            # score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
            # clap_scores.append(score.cpu().numpy())
            caption_list.append(getattr(t,'caption'))
            audio_list.append(getattr(t,'audio_path'))
            if idx % 20 == 0:
                text_embeddings = clap_model.get_text_embeddings(caption_list)# 经过了norm的embedding
                audio_embeddings = clap_model.get_audio_embeddings(audio_list, resample=True)# 这一步比较耗时，读取音频并重采样到44100
                score_mat = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
                score = score_mat.diagonal()
                clap_scores.append(score.cpu().numpy())
                # print(caption_list)
                # print(audio_list)
                # print(score)
                audio_list = []
                caption_list = []
                # print("mean:",np.mean(np.array(clap_scores).flatten()))
    return np.mean(np.array(clap_scores).flatten())

def add_clap_score_to_csv(csv_path,clap_model):
    df = pd.read_csv(csv_path,sep='\t')
    clap_scores_dict = {}
    with torch.no_grad():
        for idx,t in enumerate(tqdm(df.itertuples()),start=1): 
            text_embeddings = clap_model.get_text_embeddings([getattr(t,'caption')])# 经过了norm的embedding
            audio_embeddings = clap_model.get_audio_embeddings([getattr(t,'audio_path')], resample=True)
            score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
            clap_scores_dict[idx] = score.cpu().numpy()
    df['clap_score'] = clap_scores_dict
    df.to_csv(csv_path[:-4]+'_clap.csv',sep='\t',index=False)


if __name__ == '__main__':
    args  = parse_args()
    if args.csv_path:
        csv_path = args.csv_path
    else:
        csv_path = os.path.join(args.wavsdir,'fake_class/result.csv')
    if not os.path.exists(csv_path):
        print("result csv not exist,build for it")
        csv_path = build_csv_from_wavs(args.wavsdir)
    clap_model = CLAPWrapper(os.path.join(args.ckpt_path,'CLAP_weights_2022.pth'),os.path.join(args.ckpt_path,'config.yml'), use_cuda=True)
    clap_score = cal_score_by_csv(csv_path,clap_model)
    out = args.wavsdir if args.wavsdir else args.csv_path
    print(f"clap_score for {out} is:{clap_score}")
    print(f"clap_score for {out} is:{clap_score}")
    print(f"clap_score for {out} is:{clap_score}")
    # os.remove(csv_path)