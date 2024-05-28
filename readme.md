<div align="center">

# RLHF-V
**Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback**

<a href='https://rlhf-v.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='http://120.92.209.146:8081'><img src='https://img.shields.io/badge/Demo-Page-purple'></a>
<a href='https://arxiv.org/abs/2312.00849'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
![License](https://img.shields.io/badge/License-BSD-blue.svg)
</div>


## Brief Introduction

This repository hosts the code, data, and model weight of **RLHF-V**, a novel framework that aligns Multimodal Large Language Models (MLLMs) behavior through fine-grained correctional human feedback.

We collect <a href="https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset">fine-grained correctional feedback data</a>, which can better credit the desired behavior, by asking human annotators to correct the hallucinated segments in model responses. Benefiting from the high data efficiency, it takes only 1 hour on 8 A100 GPUs for us to reduce the hallucination rate of the base model by 34.8%. Specifically, we conduct experiments on [Muffin](https://arxiv.org/abs/2310.00653), an MLLM that has a strong ability in image understanding and reasoning which is trained on [UniMM-Chat](https://huggingface.co/datasets/Yirany/UniMM-Chat/settings).

Visit our 🏠 [project page](https://rlhf-v.github.io) and 📃 [paper](https://arxiv.org/abs/2312.00849) to explore more! And don't miss to try our interactive 🔥 [demo](http://120.92.209.146:8081)!

## 🎈News

#### 📌 Pinned

* [2024.05.28] 📃 Our RLAIF-V paper is accesible at [arxiv](https://arxiv.org/abs/2405.17220) now!
* [2024.05.20] 🎉 We introduce [RLAIF-V](https://github.com/RLHF-V/RLAIF-V), our new alignment framework that utilize open-source models for feedback generation and reach **super GPT-4V trustworthiness**. You can download the corresponding [dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) now! 

<br>

* [2024.04.11] 🔥 Our data is used in [MiniCPM-V 2.0](https://huggingface.co/openbmb/MiniCPM-V-2), an **end-side** multimodal large language model that exhibits **comparable trustworthiness with GPT-4V**!
* [2024.03.10] 📃 Our RLHF-V is accepted by [CVPR 2024](https://cvpr.thecvf.com/virtual/2024/poster/31610)!
* [2024.02.04] 🔥 [OmniLMM-12B](https://github.com/OpenBMB/OmniLMM?tab=readme-ov-file#omnilmm-12b) which is built with RLHF-V achieves the **#1 rank** among open-source models on [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench) and even **outperforms GPT-4V** on [Object HalBench](https://arxiv.org/abs/2312.00849)! The demo is avaible at [here](http://120.92.209.146:8081)!
* [2024.01.06] 🔥 A **larger, more diverse** set of fine-grained human correction data is available at [hugging face](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) now! 🔥 The newly released data has about 5.7k of fine-grained human correction data that covers the **output of more powerful models** (Qwen-VL-Chat, InstructBLIP, etc.). We also expand the image types from everyday scenes to **diverse styles and themes** (WikiArt, landmarks, scene texts, etc.).
* [2023.12.15] 🗂 We merge a new subset in our huggingface dataset! It contains an amount of **1,065 fine-grained human preference data** annotated on the outputs of **LLaVA-13B**. 
* [2023.12.04] 📃 Our paper is accesible at [arxiv](https://arxiv.org/abs/2312.00849) now. We are still working hard to improve the data **diversity** and **amount**. More high-qulity data are just on the way!

## Contents <!-- omit in toc -->

- [Dataset](#dataset)
- [RLHF-V Weights](#rlhf-v-weights)
- [Install](#install)
- [Evaluation](#evaluation)
- [RLHF-V Training](#rlhf-v-training)
- [Licenses](#licenses)
- [Acknowledgement](#acknowledgement)

## Dataset

We present the [RLHF-V-Dataset](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset), which is a human preference dataset constructed by fine-grained segment-level human corrections. In practice, we obtain a total of 1.4k annotated data that includes a diverse set of detailed description instructions and question-answering instructions.


## RLHF-V Weights

We release RLHF-V model weights on [Hugging Face](https://huggingface.co/openbmb/RLHF-V_v0).

We also provide [our SFT weights](https://huggingface.co/Yirany/RLHF-V_v0_SFT), which is the model checkpoint after finetuning Muffin on the VQAv2 dataset.

## Install

1. Install Muffin
```bash
cd RLHF-V
git clone https://github.com/thunlp/muffin

cd Muffin
# Creating conda environment
conda create -n muffin python=3.10
conda activate muffin

# Installing dependencies
pip install -e .

# Install specific version of transformers to make sure you can reproduce the experimental results in our papers
git clone --recursive git@github.com:huggingface/transformers.git
cd transformers
git checkout a92e0ad2e20ef4ce28410b5e05c5d63a5a304e65
pip install .
cd ..
```

2. Prepare training environment

Install additional packages if you need to do training.
```bash
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Note: Uncomment the following line if you have CUDA version <= 11.4
# git checkout ad11394

MAX_JOBS=8 python setup.py install
cd ..
```

3. Prepare evaluation environment

To run Object HalBench evaluation, you also need the following packages:
```bash
jsonlines
nltk==3.8.1
spacy==3.7.0

# Download and install "en_core_web_trf" for spacy
# The wheel version we use can be downloaded from
# https://github.com/explosion/spacy-models/releases/tag/en_core_web_trf-3.7.2
# run pip install en_core_web_trf-3.7.2-py3-none-any.whl
```

## Evaluation

### LLaVA Bench

Run the following script to generate, evaluate, and summarize results for LLaVA Bench:

```bash
# cd RLHF-V

bash ./script/eval/eval_muffin_llavabench.sh ./RLHF-V_weight ./results/RLHF-V {YOUR_OPENAI_API_KEY}
```

### Object HalBench

1. Prepare COCO2014 annotations

The evaluation of Object HalBench relies on the caption and segmentation annotations from the COCO2014 dataset. Please first download the COCO2014 dataset from the COCO dataset's official website.

```bash
mkdir coco2014
cd coco2014

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_trainval2014.zip
```

2. Inference, evaluation, and summarization

Please replace `{YOUR_COCO2014_ANNOTATION_DIR}` with the path for COCO2014 annotation directory(e.g. `./coco2014/annotations`), and replace `{YOUR_OPENAI_API_KEY}` with a valid OpenAI api-key.

```bash
# cd RLHF-V

bash ./script/eval_muffin_objhal.sh ./RLHF-V_weight ./results/RLHF-V {YOUR_COCO2014_ANNOTATION_DIR} {YOUR_OPENAI_API_KEY}
```

### MMHal Bench

1. Prepare MMHal Data

Please download the MMHal evaluation data [here](https://drive.google.com/file/d/1mQyAbeGgRyiVV6qjVkUI1uY_g9E-bDTH/view?usp=sharing), and save the file in `eval/data`. 

2. Run the following script to generate, evaluate, and summarize results for MMHal Bench:

```bash
# cd RLHF-V

bash ./script/eval_muffin_mmhal.sh ./RLHF-V_weight ./results/RLHF-V {YOUR_OPENAI_API_KEY}
```




## RLHF-V Training

1. Prepare environment

Please follow the instructions in the [Install](#install) section to prepare the training environment. And make sure to **upgrade to the latest code base of [Muffin](https://github.com/thunlp/muffin)**:
```
cd Muffin

git pull
pip install -e .
```

2. Prepare model checkpoint

Please download our [SFT model checkpoint](https://huggingface.co/Yirany/RLHF-V_v0_SFT/tree/main) and save it to `Muffin/RLHF-V_SFT_weight`.

3. Training

Please make sure to **upgrade to the latest code base of [Muffin](https://github.com/thunlp/muffin)**. After installing the environment of Muffin, you can train your model as follows. This script will automatically download our open-sourced training data from HuggingFace, generate logps by [our SFT model](https://huggingface.co/Yirany/RLHF-V_v0_SFT/tree/main), and do DDPO training:
```bash
cd Muffin

ref_model=./RLHF-V_SFT_weight

bash ./script/train/run_RLHFV.sh \
    ./RLHFV_checkpoints/dpo_exp \
    master \
    RLHFV \
    1.1 \
    $ref_model \
    ./RLHF-V-Dataset \
    RLHFV_SFT \
    2160 \
    360 \
    0.1 \
    False \
    True
```

## Licenses


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna, and Chat GPT. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Acknowledgement

- [Muffin](https://github.com/thunlp/muffin): the codebase we built upon.
- [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF): we utilize the MMHal-Bench data and evaluation code constructed by them.
- [Object Hallucination](https://github.com/LisaAnne/Hallucination): we refer to the CHAIR evaluation code included in the repository.


## Citation

If you find our model/code/data/paper helpful, please consider cite our papers 📝 and star us ⭐️！

```bibtex
@article{yu2023rlhf,
  title={Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback},
  author={Yu, Tianyu and Yao, Yuan and Zhang, Haoye and He, Taiwen and Han, Yifeng and Cui, Ganqu and Hu, Jinyi and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong and others},
  journal={arXiv preprint arXiv:2312.00849},
  year={2023}
}

@article{yu2024rlaifv,
  title={RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness}, 
  author={Yu, Tianyu and Zhang, Haoye and Yao, Yuan and Dang, Yunkai and Chen, Da and Lu, Xiaoman and Cui, Ganqu and He, Taiwen and Liu, Zhiyuan and Chua, Tat-Seng and Sun, Maosong},
  journal={arXiv preprint arXiv:2405.17220},
  year={2024},
}
```
