import sys
sys.path.append("/home-nfs/jjery2243542/BigVGAN")
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import torchaudio
import textgrid
import math
import re
import glob
import os
from transformers import BertConfig, BertModel, BertTokenizer
import speechbrain as sb
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import json 
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from env import AttrDict

def wav2mel_sb(signal):
    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=16000,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    return spectrogram.T

class BigVGANMel:
    def __init__(self, checkpoint_file="/share/data/speech/jjery2243542/checkpoints/BigVGAN_checkpoints/bigvgan_22khz_80band/g_05000000.zip"):
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)

    def wav2mel(self, signal):
        return mel_spectrogram(signal, self.h.n_fft, self.h.num_mels, self.h.sampling_rate, self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax).transpose(1, 2)

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

class SpeechTextDataModule(pl.LightningDataModule):
    def __init__(self, args, conf):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.conf = conf

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = SpeechTextDataset(self.args.train_id_file, self.args.data_dir, self.args.textgrid_dir, vocab_file=self.conf.data.vocab_file, default_sr=self.conf.data.sr)  
            self.val_set = SpeechTextDataset(self.args.valid_id_file, self.args.data_dir, self.args.textgrid_dir, vocab_file=self.conf.data.vocab_file, default_sr=self.conf.data.sr)
        return

    def train_dataloader(self):
        #sampler = DistributedSampler(self.train_set, shuffle=True, num_replicas=self.args.n_devices)
        #train_loader = get_dataloader(self.train_set, batch_size=self.conf.training.batch_size // self.args.n_devices, sampler=sampler)
        train_loader = get_dataloader(self.train_set, batch_size=self.conf.training.batch_size // self.args.n_devices, shuffle=True, use_wav=self.conf.data.use_wav, mel_collator=self.conf.data.mel_collator)
        return train_loader

    def val_dataloader(self):
        val_loader = get_dataloader(self.val_set, batch_size=self.conf.training.batch_size, shuffle=False, use_wav=self.conf.data.use_wav, mel_collator=self.conf.data.mel_collator)
        return val_loader

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
    def __init__(self, tokenizer, mel_collator="sb"):
        self.tokenizer = tokenizer
        self.mel_collator = mel_collator
        if self.mel_collator == "bigvgan":
            self.bigvgan_mel = BigVGANMel()

    def wav_collate_fn(self, batch):
        ids = [entry[0] for entry in batch]
        wavs = [entry[1] for entry in batch]
        phn_strings = [entry[2] for entry in batch]
        wavs, wav_len = sb.utils.data_utils.batch_pad_right(wavs)
        ret = self.tokenizer(phn_strings, return_tensors="pt", padding=True, add_special_tokens=False)
        return ids, wavs, wav_len, ret #ret["input_ids"], ret["attention_mask"]

    def mel_collate_fn(self, batch):
        ids = [entry[0] for entry in batch]
        wavs = [entry[1] for entry in batch]
        phn_strings = [entry[2] for entry in batch]
        if self.mel_collator == "sb":
            mels = [wav2mel_sb(wav) for wav in wavs]
        elif self.mel_collator == "bigvgan":
            print(wavs[0].shape)
            mels = [self.bigvgan_mel.wav2mel(wav) for wav in wavs]

        mels, mel_len = sb.utils.data_utils.batch_pad_right(mels)
        ret = self.tokenizer(phn_strings, return_tensors="pt", padding=True, add_special_tokens=False)
        return ids, mels, mel_len, ret #ret["input_ids"], ret["attention_mask"]


def get_dataloader(dataset, batch_size, mel_collator="sb", shuffle=True, sampler=None, drop_last=True, use_wav=True):
    collator = Collator(dataset.tokenizer, mel_collator=mel_collator)
    collate_fn = collator.wav_collate_fn if use_wav else collator.mel_collate_fn
    if sampler is None:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                num_workers=0, collate_fn=collate_fn, drop_last=drop_last)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                num_workers=0, collate_fn=collate_fn, drop_last=drop_last)
    return data_loader
