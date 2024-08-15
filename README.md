# Fluency Scorer

## Introduction
It's my implementation for speech fluency assessment model. 
The idea for this model is from the paper [An ASR-Free Fluency Scoring Approach with Self-Supervised Learning](<https://arxiv.org/abs/2302.09928>) (Wei Liu, Kaiqi Fu, Xiaohai Tian, Shuju Shi, Wei Li, Zejun Ma, Tan Lee) proposed in the ICASSP 2023.

These implementations are unofficial, and there might be some bugs that I missed.

But, the repo will complete as soon as possible.

## Overview of Model Structure
Here shows the main structure for this repo: 
![image](https://github.com/tangYang7/fluency_scorer/assets/114934655/e2f4d1d6-139a-40a4-a397-947a19469da4)

## Data
The SpeechOcean762 dataset used in my work is an open dataset licenced with CC BY 4.0. 
Id You have downloaded SpeechOcean762 for yourself, you can fill in your directory path to `prep_data/run.sh`.

## Directions for The Programs
### The Input Features and Labels
The input generation program are in `prep_data`.
Just run the shell script in `prep_data`.
```
cd prep_data
./run.sh
```
- The labels are fluency scores in speechocean762.
- The acoustic features are extracted by **Wav2vec_large**, where the dim is the value of 1024.
- The feats and labels files are collected in `data`.
- The cluster model is trained in `train_kmeans.py`, the model will be saved in `exp/kmeans`, which is used in fluency_scoring training later. 
- `kmeans_metric.py` is used to take a look the performance of kmeans clustering.

【**Noted**】: Force alignment result to replace the Kmeans predicted results

You can run the following programming if you want to try the Force alignment results for the replacement of cluster ID. 
```
python3 gen_ctc_force_align.py
```
If you choose this for the resource of cluster ID, you need to update the `run.sh`: make the `**cluster_pred=False**`

### Train Models for Fluency Scorer
- version for no cluster_id feature:
```
./noclu_run.sh
```
- version with cluster_id feature:
```
./run.sh
```

## Results And Performance

| Models             | Utt FLU PCC |
|--------------------|:------------:|
| GOPT (Librispeech)    |     0.756    |
| Proposed paper        |   **0.795**  |
| FluScorer+cluster_idx |     0.753    |
| Flu_TFR+cluster_idx   |     0.790    |
