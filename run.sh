#!/bin/bash
set -x

lr=1e-3
batch_size=25
hidden_dim=32
num_epochs=50
gpu_index=0
# use_device='cpu'
use_device='cuda'
depth=3
num_heads=1
SO762_dir=
load_cluster_index=True

tag=SSL_feat_fluScore
model=fluScorer
model(){
  fluScorerNoclu
  fluScorer
  flu_TFR
}

exp_dir=exp/${tag}/${lr}-${depth}-${batch_size}-${hidden_dim}-${model}/br

# repeat times
repeat_list=(0 1 2 3 4)
# repeat_list=(0)

for repeat in "${repeat_list[@]}"
do
  mkdir -p $exp_dir-${repeat}
  python3 ./train.py --lr ${lr} --exp-dir ${exp_dir}-${repeat} \
  --batch_size ${batch_size} --hidden_dim ${hidden_dim} \
  --model ${model} --n-epochs ${num_epochs} --use_device ${use_device} --gpu_index ${gpu_index} \
  --depth ${depth} --num_heads ${num_heads} --SO762_dir ${SO762_dir} --load_cluster_index ${load_cluster_index}
done
