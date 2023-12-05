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

We collect <a href="https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Hall_v0/tree/main">1.4k fine-grained correctional feedback data</a>, which can better credit the desired behavior, by asking human annotators to correct the hallucinated segments in model responses. Benefiting from the high data efficiency, it takes only 1 hour on 8 A100 GPUs for us to reduce the hallucination rate of the base model by 34.8%. Specifically, we conduct experiments on [Muffin](https://arxiv.org/abs/2310.00653), an MLLM that has a strong ability in image understanding and reasoning which is trained on [UniMM-Chat](https://huggingface.co/datasets/Yirany/UniMM-Chat/settings).

Visit our [project page](https://rlhf-v.github.io) and [paper](https://arxiv.org/abs/2312.00849) to explore more!

## News

* [12/04] Our paper is accesible at [arxiv](https://arxiv.org/abs/2312.00849) now. We are still working hard to improve the data diversity and amount, and updates are just on the way!

## Contents <!-- omit in toc -->

- [RLHF-V Data](#rlhf-v-data)
- [RLHF-V Weights](#rlhf-v-weights)
- [Install](#install)
- [Evaluation](#evaluation)
- [RLHF-V Training](#rlhf-v-training)
- [Licenses](#licenses)
- [Acknowledgement](#acknowledgement)

## Dataset

We present the [RLHF-V-Hall](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Hall_v0/tree/main) dataset, which is a human preference dataset constructed by fine-grained segment-level human corrections. In practice, we obtain a total of 1.4k annotated data that includes a diverse set of detailed description instructions and question-answering instructions.


## RLHF-V Weights

We release RLHF-V model weights on [Hugging Face](https://huggingface.co/openbmb/RLHF-V_v0).

We also provide [our SFT weights](https://huggingface.co/Yirany/RLHF-V_v0_SFT) (uploading, will be available soon), which is the model checkpoint after finetuning Muffin on the VQAv2 dataset.

## Install

Please follow the instructions in the [original repository](https://github.com/thunlp/muffin#install) to install Muffin.

To run Object HalBench evaluation, you also need the following packages:
```
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

```
# cd RLHF-V

bash ./script/eval/eval_muffin_llavabench.sh ./RLHF-V_weight ./results/RLHF-V {YOUR_OPENAI_API_KEY}
```

### Object HalBench

1. Prepare COCO2014 annotations

The evaluation of Object HalBench relies on the caption and segmentation annotations from the COCO2014 dataset. Please first download the COCO2014 dataset from the COCO dataset's official website.

```
mkdir coco2014
cd coco2014

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_trainval2014.zip
```

2. Inference, evaluation, and summarization

Please replace `{YOUR_COCO2014_ANNOTATION_DIR}` with the path for COCO2014 annotation directory(e.g. `./coco2014/annotations`), and replace `{YOUR_OPENAI_API_KEY}` with a valid OpenAI api-key.

```
# cd Muffin

bash ./script/eval/eval_muffin_objhal.sh ./RLHF-V_weight ./results/RLHF-V {YOUR_COCO2014_ANNOTATION_DIR} {YOUR_OPENAI_API_KEY}
```

3. Evaluate existing inference files

`{base_dir}` is the folder path with files for evaluation, and `{file_name}` is a shared substring in these files. Files within `{base_dir}` and its subdirectories containing `{file_name}` will be evaluated.

```
# cd Muffin

bash ./script/eval/batch_objhal_review.sh {base_dir} {file_name} {YOUR_COCO2014_ANNOTATION_DIR} {YOUR_OPENAI_API_KEY}
```

### MMHal Bench

Run the following script to generate, evaluate, and summarize results for MMHal Bench:

```
# cd RLHF-V

bash ./script/eval/eval_muffin_mmhal.sh ./RLHF-V_weight ./results/RLHF-V {YOUR_OPENAI_API_KEY}
```




## RLHF-V Training

1. Prepare training data

Please download our [RLHF-V-Hall](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Hall_v0/tree/main) dataset, and save it to the following directory:

```
Muffin/data/RLHF-V-Hall_v0
```

For training simplicity, we generate the logp values based on [RLHF-V_v0_SFT-13B](https://huggingface.co/Yirany/RLHF-V_v0_SFT/tree/main) model and provide it in our dataset in advance.

To generate logp values on your own, you can run the following script:

```
cd Muffin

bash ./script/eval/eval_muffin_inference_logp.sh ./RLHF-V_SFT_weight ./data/RLHF-V-Hall_v0 RLHF-V-Hall_v0-1401.tsv
```

2. Prepare model checkpoint

Please download our [SFT model checkpoint](https://huggingface.co/Yirany/RLHF-V_v0_SFT/tree/main).

3. Training

After installing the environment of Muffin, you can train your model as follows:
```
cd Muffin

bash ./script/train/run_RLHFV.sh ./RLHFV_checkpoints/dpo_exp master RLHFV 5.0 ./RLHF-V_SFT_weight RLHF-V-Hall_v0 1 320 40 0.5 True True
```

## Licenses


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna, and Chat GPT. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Acknowledgement

- [Muffin](https://github.com/thunlp/muffin): the codebase we built upon.
- [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF): we utilize the MMHal-Bench data and evaluation code constructed by them.
- [Object Hallucination](https://github.com/LisaAnne/Hallucination): we refer to the CHAIR evaluation code included in the repository.

If you find RLHF-V useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{2023rlhf-v,
  author      = {Tianyu Yu and Yuan Yao and Haoye Zhang and Taiwen He and Yifeng Han and Ganqu Cui and Jinyi Hu and Zhiyuan Liu and Hai-Tao Zheng and Maosong Sun and Tat-Seng Chua},
  title       = {RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback},
  journal      = {arxiv},
  year         = {2023},
}
```
