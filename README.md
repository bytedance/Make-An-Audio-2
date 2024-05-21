# Make-An-Audio 2: Temporal-Enhanced Text-to-Audio Generation
PyTorch Implementation of [Make-An-Audio 2](https://arxiv.org/abs/2305.18474)  
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.18474)
## Use pretrained model
We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://make-an-audio-2.github.io/) for audio samples.
## Quick Started
### eviroment preparation
Python==3.9
```
pip install -r requirements.txt
```

### Pretrained Models
Simply download the Diffusion weights from [Huggingface](https://huggingface.co/ByteDance/Make-An-Audio-2/blob/main/maa2.ckpt).  
Download BIGVGAN vocoder weights from [Google drive](https://drive.google.com/drive/folders/13Q9yoxrE83EahAb_7BaeawwjYE4QVGF8?usp=drive_link) \
Download CLAP text encoder weights from [Huggingface](https://huggingface.co/microsoft/msclap/blob/main/CLAP_weights_2022.pth) 

```
Download:
    maa2.ckpt and put it into ./useful_ckpts  
    BigVGAN vocoder and put it into ./useful_ckpts  
    CLAP_weights_2022.pth and put it into ./useful_ckpts/CLAP
```
The directory structure should be:
```
useful_ckpts/
├── bigvgan
│   ├── args.yml
│   └── best_netG.pt
├── CLAP
│   ├── config.yml
│   └── CLAP_weights_2022.pth
└── maa2.ckpt
```

### generate audio from text
The prompt will be parsed to structured caption by ChatGPT. you need to change the openai key following [openaikey](#jump).
```
python scripts/gen_wav.py --scale 4  --duration 10
--save_name gen_wav/test0 --prompt "A man speaks followed by a popping noise and laughter" 
```
Or you can write the structed prompt by yourself if you don't want to use ChatGPT
```
python scripts/gen_wav.py --scale 4  --duration 10 --save_name gen_wav/test0 \
--prompt "A man speaks followed by a popping noise and laughter" \
--struct_prompt "<man speaking& start>@<popping noise& mid>@<laughter& end>"
```
### generate audios of audiocaps test dataset
```
python scripts/txt2audio_for_2cap.py --scale 4  --vocoder-ckpt  useful_ckpts/bigvgan \
-b configs/text2audio-ConcatDiT-ae1dnat_Skl20d2_struct2MLPanylen.yaml \
--outdir logs/test_audiocaps_scale4  --test-dataset audiocaps  -r useful_ckpts/maa2.ckpt
```

# Train
## Data preparation
We can't provide the dataset download link for copyright issues. We provide the process code to generate melspec, count audio duration and generate structured caption.  
Before training, we need to construct the dataset information into a tsv file, which in the following format:
```tsv
name    dataset   caption	audio_path	duration	mel_path
1210764.wav	audiostock	Ping pong! Quiz Correct Answer Successful Button	.data/Audiostock/audios/1210764.wav	1.5	./data/Audiostock/audios/1210764_mel.npy
```


### generate the melspec file of audio
Assume you have already got a tsv file to link each caption to its audio_path, which mean the tsv_file have "name","audio_path","dataset" and "caption" columns in it.
To get the melspec of audio, run the following command, which will save mels in ./processed
```
python preprocess/mel_spec.py --tsv_path tmp.tsv --num_gpus 1 --max_duration 10
```

### Count audio duration
To count the duration of the audio and save duration information in tsv file, run the following command: 
```
python preprocess/add_duration.py --tsv_path tmp.tsv
```

### <span id="jump">generated structure caption from the original natural language caption</span>
Firstly you need to get an authorization token in [openai](https://openai.com/blog/openai-api), here is a [tutorial](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt). Then replace your key of variable openai_key in preprocess/n2s_by_openai.py. Run the following command to add structed caption, the tsv file with structured caption will be saved into {tsv_file_name}_struct.tsv:
```
python preprocess/n2s_by_openai.py --tsv_path tmp.tsv
```

### Place Tsv files
After generated structure caption, put the tsv with structed caption to ./data/main_spec_dir . And put tsv files without structured caption to ./data/no_struct_dir

Modify the config data.params.main_spec_dir and  data.params.main_spec_dir.other_spec_dir_path respectively in config file configs/text2audio-ConcatDiT-ae1dnat_Skl20d2_struct2MLPanylen.yaml .

## train variational autoencoder
Assume we have processed several datasets, and save the .tsv files in tsv_dir/*.tsv . Replace data.params.spec_dir_path with tsv_dir in the config file. Then we can train VAE with the following command. If you don't have 8 gpus in your machine, you can replace --gpus 0,1,...,gpu_nums
```
python main.py --base configs/research/autoencoder/autoencoder1d_kl20_natbig_r1_down2_disc2.yaml -t --gpus 0,1,2,3,4,5,6,7
```

## train latent diffsuion
After trainning VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train Diffusion model
```
python main.py --base configs/research/text2audio/text2audio-ConcatDiT-ae1dnat_Skl20d2_freezeFlananylen_drop.yaml -t  --gpus 0,1,2,3,4,5,6,7
```

## Acknowledgements
This implementation uses parts of the code from the following Github repos:  
[Latent Diffusion](https://github.com/CompVis/latent-diffusion) (for training framework, autoencoder and diffusion structure),  
[CLAP](https://github.com/microsoft/CLAP) (for text encoder),  
[NATSpeech](https://github.com/NATSpeech/NATSpeech) (for temporal transformer and melspectrogram process).



## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
@misc{huang2023makeanaudio,
      title={Make-An-Audio 2: Temporal-Enhanced Text-to-Audio Generation}, 
      author={Jiawei Huang and Yi Ren and Rongjie Huang and Dongchao Yang and Zhenhui Ye and Chen Zhang and Jinglin Liu and Xiang Yin and Zejun Ma and Zhou Zhao},
      year={2023},
      eprint={2305.18474},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

# Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.