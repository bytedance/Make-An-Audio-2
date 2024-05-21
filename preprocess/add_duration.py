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


import pandas as pd
import audioread
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse

def map_duration(tsv_withdur,tsv_toadd):# tsv_withdur 和 tsv_toadd 'name'列相同且tsv_withdur有duration信息，目标是给tsv_toadd的相同行加上duration信息。
    df1 = pd.read_csv(tsv_withdur,sep='\t')
    df2 = pd.read_csv(tsv_toadd,sep='\t')

    df = df2.merge(df1,on=['name'],suffixes=['','_y'])
    dropset = list(set(df.columns) - set(df1.columns))
    df = df.drop(dropset,axis=1)
    df.to_csv(tsv_toadd,sep='\t',index=False)
    return df

def add_duration(args):
    index,audiopath = args
    try:
        with audioread.audio_open(audiopath) as f:
            totalsec = f.duration
    except:
        totalsec = -1
    return (index,totalsec)

def add_dur2tsv(tsv_path,save_path):
    df = pd.read_csv(tsv_path,sep='\t')
    item_list = []
    for item in tqdm(df.itertuples()):
        item_list.append((item[0],getattr(item,'audio_path')))
    r = process_map(add_duration,item_list,max_workers=4,chunksize=1)
    index2dur = {}
    for index,dur in r:
        if dur == -1:
            bad_wav  = df.loc[index,'audio_path']
            print(f'bad wav:{bad_wav}')
        index2dur[index] = dur
        
    df['duration'] = df.index.map(index2dur)
    df.to_csv(save_path,sep='\t',index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--tsv_path",type=str)
    return parser.parse_args()

if __name__ == '__main__':
    pargs = parse_args()
    add_dur2tsv(pargs.tsv_path,pargs.tsv_path)
    #map_duration(tsv_withdur='tsv_maker/filter_audioset.tsv',
    #              tsv_toadd='MAA1 Dataset tsvs/V3/refilter_audioset.tsv')
