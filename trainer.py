import sys
import os
import lightning.pytorch as pl
from model import FairseqWrapper, DiscreteEncoder
from dataset import SpeechTextDataset, get_dataloader, SpeechTextDataModule
import torch
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.accelerators import find_usable_cuda_devices
import torch.nn as nn
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

class Distillation(pl.LightningModule): 
    def __init__(self, args, conf): 
        super().__init__()
        self.args = args
        self.conf = conf
        conf_dict = self.conf.toDict()
        self.save_hyperparameters(conf_dict)

        self.ssl_model = FairseqWrapper(self.conf.model.checkpoint, freeze=self.conf.model.freeze)
        self.reverse_model = DiscreteEncoder(vocab_file=self.conf.data.vocab_file, re_init=self.conf.model.re_init)
        self.layer_weight_params = nn.Parameter(torch.zeros([1, conf.model.n_layers], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=1)
        self.linears = nn.ModuleList([nn.Linear(self.conf.model.student_feature_dim, self.conf.model.teacher_feature_dim) for _ in range(self.conf.model.n_layers)])

        if self.conf.optimizer.loss_function == "MSE":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif self.conf.optimizer.loss_function == "L1":
            self.loss_fn = torch.nn.L1Loss(reduction="none") 
        elif self.conf.optimizer.loss_function == "SmoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss(reduction="none", beta=self.conf.optimizer.smooth_beta) 
        else:
            raise NotImplementedError(f"{self.conf.optimizer.loss_function} not implemented.")
        if conf.model.const_layer_weights:
            self.layer_weight_params.requires_grad = False

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
        ids, wavs, wav_len, text_inputs = batch
        ssl_feats = self.ssl_model(wavs, wav_len)
        feat_dim = ssl_feats.shape[-1]
        distilled_feats = self.reverse_model(text_inputs)

        length = min(ssl_feats.shape[2], distilled_feats.shape[2])
        ssl_feats = ssl_feats[:, :, :length]
        reversed_feats = torch.flip(ssl_feats, dims=[1])
        distilled_feats = distilled_feats[:, :, :length]

        transformed_feats = []
        for l in range(self.conf.model.n_layers):
            feats = self.linears[l](distilled_feats[:, l])
            transformed_feats.append(feats)
        transformed_feats = torch.stack(transformed_feats, dim=1)

        layer_weights = self.softmax(self.layer_weight_params)
        layer_weights = layer_weights.reshape(1, self.layer_weight_params.shape[1], 1, 1)
        loss = (layer_weights * self.loss_fn(transformed_feats, reversed_feats)).sum(dim=1)
        #loss = (layer_weights * ((distilled_feats - reversed_feats) ** 2)).sum(dim=1)
        loss_mask = text_inputs["attention_mask"].unsqueeze(dim=2)
        loss_mask = loss_mask[:, :length, :]
        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) / feat_dim

        if not return_feature:
            return loss, lr
        else:
            return loss, ssl_feats, distilled_feats
    
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

    args = parser.parse_args()

    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    if args.override is not None:
        overrides = eval(args.override)
        replace_values(conf, overrides)
    conf = munch.munchify(conf)

    distillation = Distillation(args, conf)
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
        every_n_train_steps=args.every_n_steps, 
        dirpath=ckpt_dir,
        save_last=True,
        filename="model-{step:07d}",
    )

    #tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{ckpt_dir}/logs/")
    wandb.login()
    wandb_logger = WandbLogger(project="distillation", name=ckpt_dir, resume="allow")
    wandb_logger.watch(distillation, log_freq=args.every_n_steps, log_graph=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    #trainer = pl.Trainer(accelerator="cuda", devices=args.n_devices, max_steps=conf.training.max_steps, plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)], callbacks=[checkpoint_callback, lr_monitor], val_check_interval=args.every_n_steps, check_val_every_n_epoch=None, logger=tb_logger, use_distributed_sampler=True, strategy='ddp_find_unused_parameters_true', gradient_clip_val=conf.training.clip)
    trainer = pl.Trainer(accelerator="cuda", devices=args.n_devices, max_steps=conf.training.max_steps, callbacks=[checkpoint_callback, lr_monitor], val_check_interval=args.every_n_steps, check_val_every_n_epoch=None, logger=wandb_logger, use_distributed_sampler=True, strategy='ddp_find_unused_parameters_true', gradient_clip_val=conf.training.clip)
    data = SpeechTextDataModule(args, conf)
    try:
        trainer.fit(distillation, data, ckpt_path=ckpt_path)
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
