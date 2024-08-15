SO762_dir=../speechocean762
score_json_file='scores.json'
feat_dir=../data

# python3 gen_seq_data_utt.py $SO762_dir $score_json_file
# python3 gen_seq_acoustic_feat.py $SO762_dir --feat_dir $feat_dir

# generate cluster_id by keans
python3 train_kmeans.py $SO762_dir --feat_dir $feat_dir
# python3 kmeans_metric.py $SO762_dir --feat_dir $feat_dir

# optional: try wav2vec force alignment results as cluster id
# python3 gen_ctc_force_align.py $SO762_dir --feat_dir $feat_dir
