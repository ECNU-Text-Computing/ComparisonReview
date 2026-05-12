# Comparison-Native Framework for LLM-Based Paper Evaluation

> A comparison-native framework for scientific paper evaluation with LLMs, shifting from **absolute scoring** to **relative ranking**.


## Introduction

### Key Idea

Unlike traditional evaluation pipelines that rely on **absolute scores**, CNPE emphasizes **relative judgment**:

- Which of two papers is better?
- Which pair is more informative for training?
- How can local pairwise preferences be aggregated into a meaningful global ranking?

This formulation is more robust to inconsistent score scales and better aligned with the comparative nature of human reviewing.


### Framework Pipeline

The overall pipeline consists of three stages:

#### 1. Data Construction
- Crawl paper submissions from OpenReview
- Load and preprocess review/submission data
- Compute semantic similarity between papers
- Build comparison sequences and pseudo rankings
- Construct pairwise training instances

#### 2. Training
- Supervised Fine-Tuning (SFT) cold-start for pairwise quality judgment
- Reinforcement Learning (GRPO) with comparison-based rewards

#### 3. Inference
- Generate pairwise preferences for sampled paper pairs
- Aggregate preferences into a global ranking



---

## Quick Start


### Installation

It is recommended to create a clean Python environment first.

```bash
pip install -r requirements.txt
```

### Setup

Before running the project, make sure to set the Python path:

```bash
export PYTHONPATH=.
```

You also need to configure your OpenReview account in: `config.py`


### Data Construction

#### Crawl submission data from OpenReview

```bash
python ./processor/crawl.py
```

#### Load data and construct the dataset

```bash
python ./processor/load_data.py
```

#### Compute semantic similarity to generate comparison sequences

```bash
python ./processor/similarity.py
```

#### Map rankings to pseudo scores and decisions

```bash
python ./processor/compress.py
```

#### Construct training data

```bash
python ./train/construct.py
```

### Training

#### Supervised Fine-Tuning (SFT)

```bash
export CUDA_VISIBLE_DEVICES=0
python ./train/sft.py
```

#### Serve the GRPO rollout model

```bash
export CUDA_VISIBLE_DEVICES=1
trl vllm-serve \
    --model ./model/iclr_2025/Qwen2.5-7B-Instruct-sft \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95
```


#### Run GRPO reinforcement learning

```bash
export CUDA_VISIBLE_DEVICES=0
python ./train/grpo.py
```


### Inference

#### Generate preferences from paper pairs

```bash
python main.py \
    --comparison-model "./model/iclr_2025/Qwen2.5-7B-Instruct-grpo" \
    --target-percentage 0.05 \
    --generate
```

#### Rank papers based on generated preferences

```bash
python main.py \
    --comparison-model "./model/iclr_2025/Qwen2.5-7B-Instruct-grpo" \
    --target-percentage 0.05 \
    --rank
```


---

## Others

### Citation

If you find this project useful, please consider citing the corresponding paper.

```bibtex
@article{zheng2026isolated,
  title={From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation},
  author={Zheng, Pujun and Yao, Jiacheng and Zheng, Jinquan and Gu, Chenyang and He, Guoxiu and Liu, Jiawei and Huang, Yong and Guo, Tianrui and Lu, Wei},
  journal={arXiv preprint arXiv:2603.17588},
  year={2026}
}
```