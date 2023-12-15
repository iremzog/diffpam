# DiffPam: Speeding up Photoacoustic Imaging using Diffusion Models

## Abstract

In this work, we are proposing a novel and highly adaptable DiffPam algorithm that utilizes diffusion models for speeding-up the photoacoustic imaging process. This repository is specialized version of [diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling). 

## Prerequisites
- python 3.8

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/iremzog/diffpam

cd diffpam
```

<br />

### 2) Download pretrained checkpoint

From the [link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt), download the checkpoint "256x256_diffusion_uncond.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/256x256_diffusion_uncond.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

You may use other unconditional diffusion models as well.
<br />


### 3) Set environment
Install dependencies

```
conda create -n diffpam python=3.8

conda activate diffpam

pip install -r requirements.txt
```

<br />

### 4) Inference

```
python sample_condition.py \
--model_config=configs/imagenet_model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={TASK-CONFIG};
--input={INPUT-CHOICE}
```

{INPUT-CHOICE} can be "noise", "interpolation" or "simple_sr", corresponding from scratch, bilinear and U-Net outputs respectively. The default is "noise".

<br />

### Structure of task configurations

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

simple_sr:
    model_path: 'simple_sr/models/model_ckpt_steps_5000.ckpt' # Simple SR model path

measurement:
  operator:
    name: inpainting
  mask_opt:
    mask_type: periodic
    mask_ratio: !!python/tuple [5,1] # for periodic
    mask_len_range: !!python/tuple [128, 129]  # for box
    mask_prob_range: !!python/tuple [0.8, 0.8]  # for random
    image_size: 256 

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```

## Citation
If you find our work interesting, please consider citing

```
@misc{loc2023speeding,
      title={Speeding up Photoacoustic Imaging using Diffusion Models}, 
      author={Irem Loc and Mehmet Burcin Unlu},
      year={2023},
      eprint={2312.08834},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
```
