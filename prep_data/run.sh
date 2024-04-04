python3 gen_seq_data_utt.py
python3 gen_seq_acoustic_feat.py

# use cluster id by keans
python3 train_kmeans.py
python3 kmeans_metric.py

# optional: try wav2vec force alignment results as cluster id
# python3 gen_ctc_force_align.py
