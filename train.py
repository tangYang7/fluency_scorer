# -*- coding: utf-8 -*-
# train and test the models
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchaudio import load
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import joblib
from torch.nn.utils.rnn import pad_sequence

from models import *
import argparse

class bcolors:
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def set_arg(parser):
    parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--n-epochs", type=int, default=100, help="number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--embed_dim", type=int, default=12, help="input feature embedding dimension")
    parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
    parser.add_argument("--model", type=str, default='fluScorer', help="name of the model")
    parser.add_argument("--am", type=str, default='wav2vec2.0', help="name of the acoustic models")
    parser.add_argument("--cluster_pred", type=bool, default=True, help="cluster predict in training or not")
    return parser

def convert_bin(input, num_binary=6):
    # Convert each number to its binary representation
    binary_representations = [list(map(int, bin(num)[2:].zfill(num_binary))) for num in input]
    
    # Convert to a PyTorch tensor
    tensor_2d = torch.tensor(binary_representations)
    return tensor_2d

def cluster_pred(feats, model):
    feats = feats.cpu().numpy()
    cluster_index_list = []
    for feat in feats:
        pred = model.predict(feat)
        pred_bin = convert_bin(pred)
        cluster_index_list.append(pred_bin)
    cluster_index_tensor = torch.stack(cluster_index_list, dim=0)
    return cluster_index_tensor

def draw_train_fig(train_mse_values, val_mse_values, train_corr_values, val_corr_values, epochs_list, exp_dir):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_mse_values, label='Training MSE')
    plt.plot(epochs_list, val_mse_values, label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_corr_values, label='Training Correlation')
    plt.plot(epochs_list, val_corr_values, label='Validation Correlation')
    plt.title('Training and Validation Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()

    # Annotate the PCC in the end
    ylast, xlast = val_corr_values[-1], epochs_list[-1]
    plt.text(xlast, ylast, f'{ylast:.3f}', ha='right', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{exp_dir}/train.jpg")

def train(audio_model, train_loader, test_loader, args):
    gpu_index = 0
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    train_mse_values, train_corr_values = [], []
    val_mse_values, val_corr_values = [], []
    epochs_list = []

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    audio_model = audio_model.to(device)

    if args.model == 'fluScorer':
        kmeans_model = joblib.load(f'exp/kmeans/kmeans_model.joblib')
    else:
        kmeans_model = None

    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    loss_fn = nn.MSELoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 7])
    
    while epoch < args.n_epochs:
        audio_model.train()
        for _, data in enumerate(train_loader):
            if len(data) == 3:
                audio_paths, utt_label, feats = data
            elif len(data) == 4:
                audio_paths, utt_label, feats, indexs = data
            else:
                raise ValueError("Unexpected number of elements in data")

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))
                
            if args.model == 'fluScorer' and args.cluster_pred:
                cluster_index = cluster_pred(feats, kmeans_model)
                cluster_index = cluster_index.to(device)
            else:
                cluster_index_list = []
                for index in indexs:
                    cluster_index = convert_bin(index, 6)
                    cluster_index_list.append(cluster_index)
                cluster_index_tensor = torch.stack(cluster_index_list, dim=0)
                cluster_index = cluster_index_tensor.to(device)

            feats = feats.to(device)
            if args.model == 'fluScorer':
                pred = audio_model(feats, cluster_index)
            elif args.model == 'fluScorerNoclu':
                pred = audio_model(feats)

            flu_label = torch.unsqueeze(utt_label[:, 2], 1)
            flu_label = flu_label.to(device, non_blocking=True)
            loss = loss_fn(pred, flu_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('start validation')

        # ensemble results
        # don't save prediction for the training set
        tr_mse, tr_corr = validate(audio_model, train_loader, args, -1, kmeans_model)
        te_mse, te_corr = validate(audio_model, test_loader, args, best_mse, kmeans_model)

        train_mse_values.append(tr_mse)
        train_corr_values.append(tr_corr)
        val_mse_values.append(te_mse)
        val_corr_values.append(te_corr)
        epochs_list.append(epoch)

        print('Flency: Train MSE: {:.3f}, CORR: {:.3f}'.format(tr_mse.item(), tr_corr))
        print(f'Flency: Test MSE: {te_mse.item():.3f}, {bcolors.YELLOW}CORR: {te_corr:.3f}{bcolors.ENDC}')

        result[epoch, :6] = [epoch, tr_mse, tr_corr, te_mse, te_corr, optimizer.param_groups[0]['lr']]
        print('-------------------validation finished-------------------')

        if te_mse < best_mse:
            best_mse = te_mse
            best_epoch = epoch

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if global_step > warm_up_step:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        epoch += 1

    draw_train_fig(train_mse_values, val_mse_values, train_corr_values, val_corr_values, epochs_list, exp_dir)

def validate(audio_model, val_loader, args, best_mse, kmeans_model=None):
    gpu_index = 0
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_flu, A_flu_target = [], []
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            if len(data) == 3:
                audio_paths, utt_label, feats = data
            elif len(data) == 4:
                audio_paths, utt_label, feats, indexs = data
            else:
                raise ValueError("Unexpected number of elements in data")
            
            if args.model == 'fluScorer' and args.cluster_pred:
                cluster_index = cluster_pred(feats, kmeans_model)
                cluster_index = cluster_index.to(device)
            else:
                cluster_index_list = []
                for index in indexs:
                    cluster_index = convert_bin(index, 6)
                    cluster_index_list.append(cluster_index)
                cluster_index_tensor = torch.stack(cluster_index_list, dim=0)
                cluster_index = cluster_index_tensor.to(device)

            feats = feats.to(device)
            if args.model == 'fluScorer':
                flu_score = audio_model(feats, cluster_index)
            elif args.model == 'fluScorerNoclu':
                flu_score = audio_model(feats)
            
            flu_score = flu_score.to('cpu').detach()
            flu_label = torch.unsqueeze(torch.unsqueeze(utt_label[:, 2], 1), 1)

            A_flu.append(flu_score)
            A_flu_target.append(flu_label)

        A_flu, A_flu_target = torch.cat(A_flu), torch.cat(A_flu_target)

        # get the scores
        flu_mse, flu_corr = valid_flu(A_flu, A_flu_target)

        if flu_mse < best_mse:
            print('\033[94mnew best flu mse {:.3f}, now saving predictions.\033[0m'.format(flu_mse))
            print(args.exp_dir)
            # create the directory
            if os.path.exists(args.exp_dir + '/preds') == False:
                os.mkdir(args.exp_dir + '/preds')

            # saving the phn target, only do once
            if os.path.exists(args.exp_dir + '/preds/phn_target.npy') == False:
                np.save(args.exp_dir + '/preds/flu_target.npy', A_flu_target)

            np.save(args.exp_dir + '/preds/flu_pred.npy', A_flu)

    return flu_mse, flu_corr

def valid_flu(audio_output, target):
    valid_token_pred = audio_output.view(-1).numpy()
    valid_token_target = target.view(-1).numpy()
    # print(valid_token_pred)
    # print(valid_token_target)

    valid_token_mse = np.mean((valid_token_pred - valid_token_target)**2)
    corr_matrix = np.corrcoef(valid_token_pred, valid_token_target)
    corr = corr_matrix[0, 1].item()
    return valid_token_mse, corr

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def custom_collate_fn(batch):
    # 將批次中的樣本按照 feat_x 的大小排序
    batch = sorted(batch, key=lambda x: x[2].shape[0], reverse=True)
    
    # 提取排序後的資料
    paths, utt_labels, feats = zip(*batch)

    # 將 feat_x 轉換成一個批次，這裡使用 pad_sequence 進行填充
    padded_feats = pad_sequence(feats, batch_first=True)

    return paths, torch.stack(utt_labels), padded_feats

class fluDataset(Dataset):
    def __init__(self, set, load_cluster_index=None):
        paths = load_file(f'speechocean762/{set}/wav.scp')
        # audio_list = []
        for i in range(paths.shape[0]):
            paths[i] = paths[i].split('\t')[1]

        if set == 'train':
            dataset_type = 'tr'
        elif set == 'test':
            dataset_type = 'te'
        else:
            print(f"Error: not such set called {set}")

        self.utt_label = torch.tensor(np.load(f'data/{dataset_type}_label_utt.npy'), dtype=torch.float)
        self.paths = paths
        self.load_cluster_index = load_cluster_index
        if load_cluster_index:
            with open(f'data/{dataset_type}_indexs.pkl', 'rb') as file:
                self.index = pickle.load(file)

        with open(f'data/{dataset_type}_feats.pkl', 'rb') as file:
            self.feats = pickle.load(file)

        self.utt_label = self.utt_label * 0.2

    def __len__(self):
        return self.utt_label.size(0)

    def __getitem__(self, idx):
        if self.load_cluster_index:
            # audio, utt_label, feat_x, cluster_id
            return self.paths[idx], self.utt_label[idx, :], self.feats[self.paths[idx]], self.index[self.paths[idx]]
        else:
            # audio, utt_label, feat_x
            return self.paths[idx], self.utt_label[idx, :], self.feats[self.paths[idx]]

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = set_arg(parser)
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
    args = parser.parse_args()
    if os.path.exists(args.exp_dir) == False:
        os.mkdir(args.exp_dir)
    am = args.am
    print('now train with {:s} acoustic models'.format(am))
    feat_dim = {
                'wav2vec2_base':768,
                'wav2vec2_large':1024, 
    }
    input_dim=feat_dim[am]

    if args.model == 'fluScorer':
        print('now train a fluScorer models')
        audio_model = FluencyScorer(input_dim=input_dim, embed_dim=args.embed_dim, clustering_dim=6)
    elif args.model == 'fluScorerNoclu':
        print('now train a fluScorer models <<no cluster>>')
        audio_model = FluencyScorerNoclu(input_dim=input_dim, embed_dim=args.embed_dim)

    print('Prepare datasets...')
    tr_dataset = fluDataset('train', not args.cluster_pred)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    # tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    te_dataset = fluDataset('test', not args.cluster_pred)
    te_dataloader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print('Done.')

    train(audio_model, tr_dataloader, te_dataloader, args)
