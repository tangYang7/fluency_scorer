# Generate sequence phone input and label for seq2seq models from raw Kaldi GOP features.

import numpy as np
import json
import os

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def process_feat_seq_utt(paths, utt2score):
    key_set = []
    for i in range(paths.shape[0]):
        cur_key = paths[i].split('\t')[0]
        key_set.append(cur_key)

    utt_cnt = len(list(set(key_set)))
    print('In total utterance number : ' + str(utt_cnt))
    # print(key_set)
    seq_label = np.zeros([utt_cnt, 5])

    prev_utt_id = paths[0].split('\t')[0]

    row = 0
    # for i in range(paths.shape[0]):
    for i in range(utt_cnt):
        cur_utt_id, cur_tok_path = paths[i].split('\t')[0], paths[i].split('\t')[1]
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id
        
        seq_label[row, 0] = utt2score[cur_utt_id]['accuracy']
        seq_label[row, 1] = utt2score[cur_utt_id]['completeness']
        seq_label[row, 2] = utt2score[cur_utt_id]['fluency']
        seq_label[row, 3] = utt2score[cur_utt_id]['prosodic']
        seq_label[row, 4] = utt2score[cur_utt_id]['total']

    return seq_label

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("SO762_dir", type=str, default='../speechocean762')
    parser.add_argument("score_json", type=str, default='scores.json')
    parser.add_argument("--output_dir", type=str, default='../data')
    args = parser.parse_args()

    # utt label dict
    with open(args.score_json) as f:
        utt2score = json.loads(f.read())

    dataset_path = args.SO762_dir
    output_dir = args.output_dir
    if not os.path.exists(dataset_path):
        # download the dataset in dataset_path
        os.system('wget https://www.openslr.org/resources/101/speechocean762.tar.gz')
        os.system(f'tar -xvzf speechocean762.tar.gz -C {dataset_path}')        
        
    # sequencialize training data
    tr_paths = load_file(f'{dataset_path}/train/wav.scp')
    tr_label = process_feat_seq_utt(tr_paths, utt2score)
    print(tr_label.shape)
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/tr_label_utt.npy', tr_label)

    te_paths = load_file(f'{dataset_path}/test/wav.scp')
    te_label = process_feat_seq_utt(te_paths, utt2score)
    print(te_label.shape)
    os.makedirs("../data", exist_ok=True)
    np.save(f'{output_dir}/te_label_utt.npy', te_label)
