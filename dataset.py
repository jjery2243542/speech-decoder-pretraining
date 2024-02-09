from torch.utils.data import Dataset, DataLoader
import torchaudio
import textgrid
import math
import re
import glob
import os
from transformers import BertConfig, BertModel, BertTokenizer
import speechbrain as sb

def remove_digit_part(input_string):
    # Use regular expression to remove digits
    result = re.sub(r'\d', '', input_string)
    return result

def tg2phn(tg, frame_time=0.02):
    dup_phns = []
    phns = []
    durations = []
    # index 1 is phones
    for phn_interval in tg[1]:
        phn = remove_digit_part(phn_interval.mark)
        start = math.floor(phn_interval.minTime / frame_time)
        end = math.floor(phn_interval.maxTime / frame_time)
        dup_phns.extend([phn for _ in range(end - start)])
        phns.append(phn)
        durations.append(end - start)
    return dup_phns, phns, durations   

class SpeechTextDataset(Dataset):
    def __init__(self, id_file, data_dir, textgrid_dir, vocab_file, default_sr=16000):

        self.default_sr = default_sr
        self.data_dir = data_dir
        self.textgrid_dir = textgrid_dir
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.tokenizer.pad_token_id = 0

        self.ids = []
        with open(id_file) as f:
            for line in f:
                id = line.strip()
                self.ids.append(id)

        tg_file_list = [f"{id}.TextGrid" for id in self.ids]
        #self.ids = [path.split("/")[-1].replace(".TextGrid", "") for path in tg_file_list]
        self.tg_list = [textgrid.TextGrid.fromFile(os.path.join(textgrid_dir, path)) for path in tg_file_list]
        self.phn_list = [tg2phn(tg) for tg in self.tg_list]
        self.wav_list = [id.strip().split("/")[-1].replace("TextGrid", "wav") for id in tg_file_list]
        self.wav_list = [os.path.join(self.data_dir, filename) for filename in self.wav_list]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        id = self.ids[index]
        wav, sr = torchaudio.load(self.wav_list[index])
        if len(wav.shape) == 2:
            wav = wav[0]
        if sr != self.default_sr:
            wav = torchaudio.functional.resample(wav, sr, self.default_sr)
        dup_phns, phns, durations = self.phn_list[index]
        dup_phn_strings = " ".join(dup_phns)
        return id, wav, dup_phn_strings, phns, durations

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        ids = [entry[0] for entry in batch]
        wavs = [entry[1] for entry in batch]
        phn_strings = [entry[2] for entry in batch]
        wavs, wav_len = sb.utils.data_utils.batch_pad_right(wavs)
        ret = self.tokenizer(phn_strings, return_tensors="pt", padding=True, add_special_tokens=False)
        return ids, wavs, wav_len, ret #ret["input_ids"], ret["attention_mask"]


def get_dataloader(dataset, batch_size, shuffle=True, sampler=None, drop_last=True):
    collator = Collator(dataset.tokenizer)
    if sampler is None:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                num_workers=0, collate_fn=collator.collate_fn, drop_last=drop_last)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                num_workers=0, collate_fn=collator.collate_fn, drop_last=drop_last)
    return data_loader
