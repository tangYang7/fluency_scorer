import pickle

# 指定保存文件的路径
file_path = '../data/tr_feats.pkl'

# 使用 pickle.load() 从文件中读取张量
with open(file_path, 'rb') as file:
    loaded_tensor = pickle.load(file)
print(type(loaded_tensor))
print('Loaded tensor:', loaded_tensor)


file_path = '../data/tr_cluster_index.pkl'

# 使用 pickle.load() 从文件中读取张量
with open(file_path, 'rb') as file:
    loaded_tensor = pickle.load(file)

# print('Loaded tensor:', loaded_tensor)


file_path = '../data/cluster_centers.pkl'

# 使用 pickle.load() 从文件中读取张量
with open(file_path, 'rb') as file:
    loaded_tensor = pickle.load(file)
# print('Loaded tensor:', loaded_tensor)