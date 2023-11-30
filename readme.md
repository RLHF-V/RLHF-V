<div align="center">

# RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback
<a href='https://rlhf-v.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'> </a>
<a href='http://120.92.209.146:8081'><img src='https://img.shields.io/badge/Demo-Page-purple'></a>
<a href='https://github.com/RLHF-V/RLHF-V/blob/main/assets/RLHF-V.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
![License](https://img.shields.io/badge/License-BSD-blue.svg)
</div>

This repository hosts the code, data and model weight of **RLHF-V**, a novel framework that aligns Multimodal Large Language Models (MLLMs) behavior through fine-grained correctional human feedback.

## Brief Introduction

We collect <a href="https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Hall_v0/tree/main">1.4k fine-grained correctional feedback data</a>, which can better credit the desired behavior, by asking human annotators to correct the hallucinated segments in model responses.

Benefits from the high data efficiency, it takes only 1 hour on 8 A100 GPUs for us to reduce the hallucination rate of the base model by 34.8%. Specifically, we conduct experiments on [Muffin](https://arxiv.org/abs/2310.00653), a MLLM that has a strong ability in image understanding and reasoning which is trained on [UniMM-Chat](https://huggingface.co/datasets/Yirany/UniMM-Chat/settings).

Visit our [project page](https://rlhf-v.github.io) and [paper](assets/RLHF-V.pdf) to explore more!


## Contents <!-- omit in toc -->

- [RLHF-V Data](#rlhf-v-data)
- [RLHF-V Weights](#rlhf-v-weights)
- [Install](#install)
- [Evaluation](#evaluation)
- [RLHF-V Training](#rlhf-v-training)
- [Licenses](#licenses)
- [Acknowledgement](#acknowledgement)

## RLHF-V Data

We present the RLHF-V-Hall dataset, which is a human preference dataset constructed by fine-grained segment-level human corrections. In practice, we obtain a total of 1.4k annotated data that includes a diverse set of detailed description instructions and question answering instructions.

You can download our RLHF-V-Hall dataset from [RLHF-V-Hall_v0](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Hall_v0/tree/main).

## RLHF-V Weights

We release RLHF-V model weights on [Hugging Face](https://huggingface.co/openbmb/RLHF-V_v0).

We also provide the [our SFT weights](https://huggingface.co/Yirany/MuffinQA/tree/main) (uploading, will be available soon), which is the model checkpoint after finetuning Muffin on VQAv2 dataset.

## Install

Please follow the instructions in the [original repository](https://github.com/thunlp/muffin#install) to install Muffin.

## Evaluation

The evaluation process is identical to the Muffin project, simply follow the instructions in [Muffin evaluation](https://github.com/thunlp/Muffin#evaluation) for deployment.

## RLHF-V Training

1. Prepare training data

Please download our [RLHF-V-Hall_v0 dataset](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Hall_v0/tree/main), and save it to the following directory:

```
./data/RLHF-V-Hall_v0
```

2. Prepare model checkpoint

Please download our MuffinQA model checkpoint.

3. Training

After installing the environment of Muffin, you can train your model as follows:
```
cd Muffin


bash ./script/train/run_RLHFV.sh ../RLHFV_checkpoints/dpo_exp master RLHFV 5.0 ../RLHF-V_SFT_weight checkpoint dpo_cvpr_docrp_vqa 1 320 40 True True
```

## Licenses


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and Chat GPT. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Acknowledgement

- [Muffin](https://github.com/thunlp/muffin): the codebase we built upon.

If you find RLHF-V useful for your your research and applications, please cite using this BibTeX:
```bibtex
@article{2023rlhf-v,
  author      = {Tianyu Yu and Yuan Yao and Haoye Zhang and Taiwen He and Yifeng Han and Ganqu Cui and Jinyi Hu and Zhiyuan Liu and Hai-Tao Zheng and Maosong Sun},
  title       = {RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback},
  journal      = {arxiv},
  year         = {2023},
}
```
