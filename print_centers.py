import os
from pprint import pprint
import pickle

with open(f'data/cluster_centers.pkl', 'rb') as file:
    feats = pickle.load(file)

fig_path = f'exp/tsne_fig'
if os.path.exists(fig_path) == False:
    os.mkdir(fig_path)

with open(f'{fig_path}/cluster_centers.txt', 'w') as file:
    pprint(feats, stream=file)

print(f'Mission Completed. Check the {fig_path}/cluster_cetner.txt')
