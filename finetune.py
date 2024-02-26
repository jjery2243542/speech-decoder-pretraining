import sys
import os
import lightning.pytorch as pl
from model import FairseqWrapper, DiscreteEncoder, UpsampleModule
from dataset import SpeechTextDataset, get_dataloader
import torch
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.accelerators import find_usable_cuda_devices
import torch.nn as nn
import torch.nn.functional as F
import argparse 
import wandb
import yaml
import munch
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from scheduler import get_cosine_schedule_with_warmup
from lightning.pytorch.callbacks import LearningRateMonitor
import glob
from utils import extract_number, replace_values
import json
from trainer import SpeechTextDataModule
import speechbrain as sb
from speechbrain.nnet.losses import compute_length_mask

class TTSFinetuning(pl.LightningModule): 
    def __init__(self, args, conf): 
        super().__init__()
        self.args = args
        self.conf = conf
        conf_dict = self.conf.toDict()
        self.save_hyperparameters(conf_dict)

        self.reverse_model = DiscreteEncoder(vocab_file=self.conf.data.vocab_file, re_init=self.conf.model.re_init)
        self.tts_head = nn.Linear(self.conf.model.student_feature_dim, self.conf.model.mel_dim)
        self.upsample = UpsampleModule(self.conf.model.student_feature_dim, factor=self.conf.model.scale_factor)

        if self.conf.optimizer.loss_function == "MSE":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif self.conf.optimizer.loss_function == "L1":
            self.loss_fn = torch.nn.L1Loss(reduction="none") 
        elif self.conf.optimizer.loss_function == "SmoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss(reduction="none", beta=self.conf.optimizer.beta) 
        else:
            raise NotImplementedError(f"{self.conf.optimizer.loss_function} not implemented.")

    def configure_optimizers(self): 
        optimizer_class = getattr(torch.optim, self.conf.optimizer.name)
        if self.conf.optimizer.name == "AdamW":
            opt = optimizer_class(self.parameters(), lr=self.conf.optimizer.lr, weight_decay=self.conf.optimizer.weight_decay)
        else:
            opt = optimizer_class(self.parameters(), lr=self.conf.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=self.conf.training.num_warmup_steps, num_training_steps=self.conf.training.max_steps)
        lr_scheduler_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
        return lr_scheduler_config
    
    def forward(self, batch, return_feature=False):
        if self._trainer is not None:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        ids, mels, mel_len, text_inputs = batch
        print(mels.shape)
        # only take the last layer
        feats = self.reverse_model(text_inputs)[:, -1]
        print(feats.shape)
        feats = self.upsample(feats)
        print(feats.shape)

        if not return_feature:
            length = min(mels.shape[1], feats.shape[1])
            mels = mels[:, :length]
            feats = feats[:, :length]

        predicted_mels = self.tts_head(feats)

        if not return_feature:
            feat_dim = mels.shape[-1]
            loss = self.loss_fn(predicted_mels, mels)
            # use one of the feature as example 
            loss_mask = compute_length_mask(mels[:, :, 0], length=mel_len)[:, :length].unsqueeze(dim=-1)
            loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) / feat_dim

        if not return_feature:
            return loss, lr
        else:
            return predicted_mels
    
    def training_step(self, batch, batch_idx): 
        loss, lr = self.forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/lr", lr, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx): 
        loss, _ = self.forward(batch)
        self.log("validation/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
    parser.add_argument('--every_n_steps', help='every n steps, do validation and checkpointing', default=5000, type=int)
    parser.add_argument('--override', help='override the hyperparameters in conf', default=None, type=str)
    parser.add_argument('--ckpt_path', help='path to checkpoint to load', default=None, type=str)

    args = parser.parse_args()

    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    if args.override is not None:
        overrides = eval(args.override)
        replace_values(conf, overrides)
    conf = munch.munchify(conf)

    finetuning = TTSFinetuning(args, conf)
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        state_dict = finetuning.reverse_model.state_dict()
        finetuning.reverse_model.load_state_dict(state_dict, strict=True)

    conf_path = args.conf.split("/")[-1].replace(".yaml", "")
    ckpt_dir = os.path.join(args.save_path, conf_path)
    ckpts = sorted(glob.glob(f"{ckpt_dir}/*.ckpt"), key=extract_number)
    ckpt_path = None if len(ckpts) == 0 else ckpts[-1]
    if ckpt_path is not None:
        print(f"loading from {ckpt_path}")

    # save conf to ckpt_dir
    conf_dict = conf.toDict()
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(os.path.join(ckpt_dir, "conf.yaml"), "w") as f:
        yaml.dump(conf_dict, f)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="step",
        mode="max",
        save_last=True,
        every_n_train_steps=args.every_n_steps, 
        dirpath=ckpt_dir,
        filename="model-{step:07d}",
    )

    #tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{ckpt_dir}/logs/")
    wandb.login()
    wandb_logger = WandbLogger(project="distillation_ft", name=ckpt_dir, resume="allow")
    wandb_logger.watch(finetuning, log_freq=args.every_n_steps, log_graph=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    #trainer = pl.Trainer(accelerator="cuda", devices=args.n_devices, max_steps=conf.training.max_steps, plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)], callbacks=[checkpoint_callback, lr_monitor], val_check_interval=args.every_n_steps, check_val_every_n_epoch=None, logger=tb_logger, use_distributed_sampler=True, strategy='ddp_find_unused_parameters_true', gradient_clip_val=conf.training.clip)
    trainer = pl.Trainer(accelerator="cuda", devices=args.n_devices, max_steps=conf.training.max_steps, callbacks=[checkpoint_callback, lr_monitor], val_check_interval=args.every_n_steps, check_val_every_n_epoch=None, logger=wandb_logger, use_distributed_sampler=True, strategy='ddp_find_unused_parameters_true', gradient_clip_val=conf.training.clip)
    data = SpeechTextDataModule(args, conf)
    try:
        trainer.fit(finetuning, data, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
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
