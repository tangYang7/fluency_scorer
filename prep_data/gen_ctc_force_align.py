import logging
import os
import sys
import torch
import torchaudio
from transformers import Wav2Vec2Model, HubertModel
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
from tqdm import tqdm
import pickle

from force_align import get_trellis, backtrack 

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

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("main")

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

class fluDataset(Dataset):
    def __init__(self, types):
        paths = load_file(f'../speechocean762/{types}/wav.scp')
        texts = load_file(f'../speechocean762/{types}/text')
        for i in range(paths.shape[0]):
            paths[i] = paths[i].split('\t')[1]
            texts[i] = texts[i].split('\t')[1]
        self.paths, self.texts = paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # audio, text
        return self.paths[idx], self.texts[idx]


def get_align_index(dataLoader, dataset_type):
    extract_feat_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    labels = bundle.get_labels()
    for i, (audio_paths, texts) in tqdm(enumerate(dataLoader), total=len(dataLoader)):
        # load waveform
        audio_list = []
        for audio_path in audio_paths:
            waveform, sample_rate = torchaudio.load(f"../speechocean762/{audio_path}")
            audio_list.append(waveform.clone().detach().float())

        audio = torch.stack(audio_list, dim=0)
        audio = audio.to(device)
        audio = audio.view(audio.size(0), -1)

        model = bundle.get_model().to(device)
        with torch.inference_mode():
            emissions, _ = model(audio)
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        # example: "|I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"
        # We convert the " " into "|", enclose the transcript with space tokens, which represent SOS and EOS.
        transcript = texts[0].replace(" ", "|")
        transcript = f"|{transcript}|"
        dictionary = {c: i for i, c in enumerate(labels)}

        tokens = [dictionary[c] for c in transcript]
        # print(list(zip(transcript, tokens)))

        trellis = get_trellis(emission, tokens)
        path = backtrack(trellis, emission, tokens)

        reults = []
        for x, _ in enumerate(path):
            # print(transcript[path[x].token_index])
            reults.append(dictionary[transcript[path[x].token_index]])

        results = torch.tensor(reults)
        extract_feat_list.append(results)

    # save the extracted feature to a dict
    saved_tensor_dict = {}
    for j, (paths, texts) in enumerate(dataLoader):
        for path in paths:
            if path not in saved_tensor_dict:
                saved_tensor_dict[path] = extract_feat_list[j]
                # print(saved_tensor_dict[path])

    with open(f'../data/{dataset_type}_indexs.pkl', 'wb') as file:
        pickle.dump(saved_tensor_dict, file)

    return 

def main(
    modelName,
    gpu,
    SSL_model,
):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    batch_size = 1
    tr_dataset = fluDataset('train')
    te_dataset = fluDataset('test')
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
    te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

    get_align_index(tr_dataloader, 'tr')
    get_align_index(te_dataloader, 'te')

    logger.info(f"{bcolors.YELLOW}finished successfully{bcolors.ENDC}")
    return 

if __name__ == "__main__":
    import argparse

    modelName = f"facebook/wav2vec2-large-lv60"
    # modelName = f"facebook/hubert-large-ls960-ft"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelName", default=modelName, type=str, help="the pretrained model name"
    )
    parser.add_argument("--gpu", default=True, type=bool, help="use gpu or not")
    parser.add_argument("--SSL_model", default="wav2vec2", type=str, help="Now only support wav2vec2")
    args = parser.parse_args()
    logging.info(str(args))

    main(**vars(args))
