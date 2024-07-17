# DNI: Dilutional Noise Initialization for Diffusion Video Editing, ECCV 2024

[![arXiv](https://img.shields.io/badge/arXiv-FRAG-b31b1b.svg)](https://arxiv.org/abs/2307.10373) 


**DNI** is a framework that enhances the effectiveness of edited videos by mitigating rigidity from original input video visual structures.

[//]: # (### Abstract)
>Text-based diffusion video editing systems have been successful in performing edits with high fidelity and textual alignment. However, this success is limited to rigid-type editing such as style transfer and object overlay, while preserving the original structure of the input video. This limitation stems from an initial latent noise employed in diffusion video editing systems. The diffusion video editing systems prepare initial latent noise to edit by gradually infusing Gaussian noise onto the input video. However, we observed that the visual structure of the input video still persists within this initial latent noise, thereby restricting non-rigid editing such as motion change necessitating structural modifications. To this end, this paper proposes Dilutional Noise Initialization (DNI) framework which enables editing systems to perform precise and dynamic modification including non-rigid editing. DNI introduces a concept of `noise dilution' which adds further noise to the latent noise in the region to be edited to soften the structural rigidity imposed by input video, resulting in more effective edits closer to the target prompt. Extensive experiments demonstrate the effectiveness of the DNI framework.

## Environment for tuning free based DNI
```
conda create -n dni python=3.10
conda activate dni
cd Tuning_free_DNI
pip install -r requirements.txt
```

## DDIM inversion

Preprocess you video by running using the following command:
```
python ddim_inversion.py
```
## Editing
```
python dni.py
```

## Environment for tuning based DNI
```
conda create -n dni_tuning python=3.10
conda activate dni_tuning
cd tuning_dni
pip install -r requirements.txt
```

## model tuning

Preprocess you video by running using the following command:
```
python dni_tuning.py
```
## Editing
```
python dni.py
```

## Acknowledgement

This code is implemented on top of following contributions: [TAV](https://github.com/showlab/Tune-A-Video), [TokenFlow](https://github.com/omerbt/TokenFlow), [HuggingFace](https://github.com/huggingface/transformers), [FLATTEN](https://github.com/yrcong/flatten), [FateZero](https://github.com/ChenyangQiQi/FateZero), [Prompt-to-prompt](https://github.com/google/prompt-to-prompt) 

We thank the authors for open-sourcing these great projects and papers!

This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).


