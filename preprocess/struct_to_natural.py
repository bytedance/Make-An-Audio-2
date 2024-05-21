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
import requests
import traceback
openai_key = 'your openai key here'
def get_natural(caplist):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'{openai_key}',
    }

    json_data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {
                'role': 'user',
                'content':f'I want you to generate the text discribing the sound scene based on the structed input:\
                Question: <running water& all>@<birds chriping& start>@<cow footsteps& mid>@<birds flying away& end>\
                Answer: A bird sings on the river, a cow passes by then the bird flies away \
                Question: <variety cellphone ringing tones& start>@<loud explosion& end>@<fire crackling& end>@<truck engine idle& end>\
                Answer: cellphone ringing a variety of tones followed by a loud explosion and fire crackling as a truck engine runs idle\
                Question: <train passing& all>@<short honks three times& end> \
                Answer: Train passing before short honks three times\
                Question: <Applause Clapping& all>@<Gong& start>@<Steel guitar and slide guitar& mid>\
                Answer: Gong sounds start followed by steel guitar and slide guitar with applause and clapping in the background.\
                Question: <Basketball bounce& all>@<Whoop& all>\
                Answer: Whoop and basketball bounce\
                All indicates that sound exists in the whole scene. Start, mid, end indicates the time period the sound appear.\
                Please answer the following questions, each answer should be start from a newline: \
                0. {caplist[0]} \
                1. {caplist[1]} \
                2. {caplist[2]} \
                3. {caplist[3]} \
                4. {caplist[4]} \
                5. {caplist[5]} \
                6. {caplist[6]} \
                7. {caplist[7]} \
                Answer:',
            },
        ],
        'temperature': 0.3,
    }
                # 8. {caplist[8]} \
                # 9. {caplist[9]} \
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)
    return eval(response.content)['choices'][0]["message"]["content"]



if __name__ == '__main__':
    cap_list_num = 8
    cap_list = []
    tsv_path = './bad_df.tsv'
    ori_df = pd.read_csv(tsv_path,sep='\t')
    index = 0
    end = len(ori_df)
    f = open('gen_natural.txt','w')
    while index < end - 1:
        try:
            df = ori_df.iloc[index:end]
            for t in df.itertuples():
                index = t[0]
                strcut_cap = getattr(t,'caption')
                cap_list.append(strcut_cap)
                if len(cap_list) == cap_list_num:
                    gen_captions = get_natural(cap_list)
                    gen_captions = gen_captions.split('\n')
                    print(gen_captions)
                    for i in range(cap_list_num):
                        f.write(f'{index - cap_list_num + 1 + i}\t{gen_captions[i]}\n')
                    f.flush()
                    cap_list = []
        except Exception as e:
            print(e)# 报错信息
            print(traceback.format_exc())
            f.flush()
            cap_list = []
    f.close()