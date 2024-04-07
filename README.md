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
You don't have to prepare SpeechOcean762 for yourself, but you need to run the code in `prep_data` first.

## Directions for The Programs
### The Input Features and Labels
The input generation program are in `prep_data`.
- run the shell script in `prep_data`.
```
cd prep/data
python3 gen_seq_data_utt.py
python3 gen_seq_acoustic_feat.py
python3 train_kmeans.py
python3 kmeans_metric.py
```
- The labels are utterance-level scores, which the **fluency score is `utt_label[: 2]`**.
- The acoustic features are extracted by **Wav2vec_large**, where the dim is the value of 1024.
- The cluster also can be prepare in `gen_seq_acoustic_feat.py`, but my program got killed somehow maybe out of memory, so I handle them in seperated programs. 
- The feats and labels files are collected in `data`.


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

|                    | Utt Pros PCC |
|--------------------|:------------:|
| Wav2vec_feat              |     0.413    |
| Wav2vec_feat+cluster_idx  |     0.471    |
| GOPT (Librispeech)        |     0.756    |
| Proposed paper            |   **0.795**  |

### How to explain the results?
- **The model is a pretrained model?**

  In the proposed paper by ByteDance, the fluency scorer model might be a pretrained model from ByteRead or ByteQA, then fine-tuned on Speechocean762. This results in a performance gap, as ByteRead and ByteQA offer more abundant corpora for training compared to Speechocean762.
- **Unresolved issues**

  There are some unresolved issues that fail to demonstrate the effect of masking in BiLSTM, for instance. This requires further investigation and experimentation for confirmation.
