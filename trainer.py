import lightning.pytorch as pl
from model import FairseqWrapper, DiscreteEncoder
from dataset import SpeechTextDataset, get_dataloader
import torch
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.accelerators import find_usable_cuda_devices
import torch.nn as nn
import os
import argparse  
import yaml
import munch
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal


class Distillation(pl.LightningModule): 
    def __init__(self, args, conf): 
        super().__init__()
        self.args = args
        self.conf = conf
        self.ssl_model = FairseqWrapper(self.conf.data.checkpoint, freeze=self.conf.model.freeze)
        self.reverse_model = DiscreteEncoder(vocab_file=self.conf.data.vocab_file)
        self.layer_weight_params = nn.Parameter(torch.zeros([1, conf.model.n_layers], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=1)
        if conf.model.const_layer_weights:
            self.layer_weight_params.requires_grad = False

    def configure_optimizers(self): 
        optimizer_class = getattr(torch.optim, self.conf.optimizer.name)
        return optimizer_class(self.parameters(), lr=self.conf.optimizer.lr) 
    
    #def test_dataloader(self):
    #    pass 

    def prepare_data(self): 
        self.train_set = SpeechTextDataset(self.args.train_id_file, self.args.data_dir, self.args.textgrid_dir, vocab_file=self.conf.data.vocab_file)  
        self.val_set = SpeechTextDataset(self.args.valid_id_file, self.args.data_dir, self.args.textgrid_dir, vocab_file=self.conf.data.vocab_file) 
        return

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_set, shuffle=True, num_replicas=self.args.n_devices)
        train_loader = get_dataloader(self.train_set, batch_size=self.conf.training.batch_size // self.args.n_devices, sampler=sampler)
        return train_loader

    def val_dataloader(self):
        val_loader = get_dataloader(self.val_set, batch_size=self.conf.training.batch_size, shuffle=False)
        return val_loader

    def forward(self, batch):
        ids, wavs, wav_len, text_inputs = batch
        ssl_feats = self.ssl_model(wavs, wav_len)
        feat_dim = ssl_feats.shape[-1]
        #reversed_feats = ssl_feats[:, ::-1, :, :]
        distilled_feats = self.reverse_model(text_inputs)

        length = min(ssl_feats.shape[2], distilled_feats.shape[2])
        ssl_feats = ssl_feats[:, :, :length]
        distilled_feats = distilled_feats[:, :, :length]
        reversed_feats = torch.flip(ssl_feats, dims=[1])
        layer_weights = self.softmax(self.layer_weight_params)
        layer_weights = layer_weights.reshape(1, self.layer_weight_params.shape[1], 1, 1)
        loss = (layer_weights * ((distilled_feats - reversed_feats) ** 2)).sum(dim=1)
        loss_mask = text_inputs["attention_mask"].unsqueeze(dim=2)
        loss_mask = loss_mask[:, :length, :]
        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) / feat_dim 
        return loss
    
    def training_step(self, batch, batch_idx): 
        loss = self.forward(batch)
        self.log("mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx): 
        loss = self.forward(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    #def validation_epoch_end(self, outputs): 
    #    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

    #    tensorboard_logs = {'val_loss': avg_loss}
    #    return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}
    
    #def test_step(self, batch, batch_idx):
    #    pass


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', help='Path to dataset folder for LJSpeech', default="/share/data/speech/jjery2243542/data/LJSpeech-1.1/wavs")
    parser.add_argument('--textgrid_dir', help='Path to textgrid folder for LJSpeech', default="/share/data/speech/jjery2243542/alignment/LJSpeech/TextGrid/LJSpeech")
    parser.add_argument('--conf', help='Path to config file', default="conf/LJ_speech_text_hubert_base.yaml")
    parser.add_argument('--train_id_file', help='Path to training dataset ids for LJSpeech', default="/share/data/speech/jjery2243542/data/LJSpeech/train.txt")
    parser.add_argument('--valid_id_file', help='Path to validation dataset ids for LJSpeech', default="/share/data/speech/jjery2243542/data/LJSpeech/valid.txt")
    parser.add_argument('--save_path', help='Path to save checkpoints', default="/home-nfs/jjery2243542/reverse_feature_distillation/checkpoints")
    parser.add_argument('--n_devices', help='number of gpus', default=1, type=int)

    args = parser.parse_args()

    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    conf = munch.munchify(conf)

    distillation = Distillation(args, conf)
    trainer = pl.Trainer(default_root_dir=args.save_path, max_epochs=conf.training.epochs, strategy='ddp_find_unused_parameters_true', plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)])
    trainer.fit(distillation)

if __name__ == "__main__":
    main()
    #args = placeholder()
    #args.save_path = "/share/data/speech/jjery2243542/checkpoints/hubert/hubert_base_ls960.pt"
    #args.vocab_file = "/home-nfs/jjery2243542/reverse_feature_distillation/vocab_files/phn_vocab.txt"
    #args.train_id_file = "/share/data/speech/jjery2243542/data/LJSpeech/train.txt"
    #args.val_id_file = "/share/data/speech/jjery2243542/data/LJSpeech/train.txt"
    #args.data_dir = "/share/data/speech/jjery2243542/data/LJSpeech-1.1/wavs"
    #args.textgrid_dir = "/share/data/speech/jjery2243542/alignment/LJSpeech/TextGrid/LJSpeech"
    #conf = placeholder()
    #args.save_path = "/share/data/speech/jjery2243542/checkpoints/hubert/hubert_base_ls960.pt"
    #args.vocab_file = "/home-nfs/jjery2243542/reverse_feature_distillation/vocab_files/phn_vocab.txt"
    #conf.lr = 1e-4
    #conf.freeze = True
    #conf.n_layers = 12
    #conf.const_layer_weights = True
    #conf.batch_size = 8
    #distillation = Distillation(args, conf)
    #trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    #trainer.fit(distillation)
