
import os
import yaml
from collections import OrderedDict

import numpy as np
from scipy.stats import pearsonr

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection, MeanSquaredError, PearsonCorrCoef
from pytorch_lightning.utilities import cli as pl_cli
from pytorch_lightning.utilities.cli import LightningCLI
import pytorch_lightning as pl
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import matplotlib.pyplot as plt
import seaborn as sns

from dali_input import tfrecord_pipeline

# ref: https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]
    
class permute(nn.Module):
    def forward(self,x):
        return torch.permute(x, (0, 2, 1))

class squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)

class BichromDataLoaderHook(pl.LightningModule):
    """
    Define universal things:
    1. Dataloader
    2. Hooks
    """
    def __init__(self, data_config, batch_size=512, target_vlog=True):
        super().__init__()
        self.batch_size = batch_size
        self.target_vlog = target_vlog

        config = yaml.safe_load(open(data_config, "r"))
        self.train_path = config['train_bichrom']
        self.val_path = config['val']
        self.test_path = config['test']
        self.scaler_means = self.train_path['scaler_mean']
        self.scaler_vars = self.train_path['scaler_var']
        
        self.metrics = MetricCollection([MeanSquaredError(), PearsonCorrCoef()])
        self.test_hpmetrics0 = self.metrics.clone(prefix="hp/test_label0_")
        self.test_hpmetrics1 = self.metrics.clone(prefix="hp/test_label1_")

        self.example_input_array = [torch.zeros(512, 4, 500).index_fill_(1, torch.tensor(2), 1), torch.ones(512, 12, 500)]

    def vlog(self, tensor):
        """
        log(tensor+1) operation
        """
        return torch.log(torch.add(tensor, 1))

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_test_start(self):
        self.logger.log_hyperparams(self.hparams, {i:0 for i in list(self.test_hpmetrics0.keys()) + list(self.test_hpmetrics1.keys())})

    def training_step(self, batch, batch_idx):
        # define train loop
        seq, chroms, y, label = batch
        y_hat = self(seq, chroms)

        # compute prediction and loss
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # define validation loop
        seq, chroms, y, label = batch
        y_hat = self(seq, chroms)

        # compute prediction and loss
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return {'val_loss': val_loss, 'pred': y_hat, 'true': y}

    def test_step(self, batch, batch_idx):
        # define test
        seq, chroms, y, label = batch
        y_hat = self(seq, chroms)

        # compute prediction and loss
        test_loss = F.mse_loss(y_hat, y)

        # compute metrics
        if (label==0).sum()>1: self.test_hpmetrics0(y_hat[label==0], y[label==0])
        if (label==1).sum()>1: self.test_hpmetrics1(y_hat[label==1], y[label==1])

        return {'test_loss': test_loss, 'pred': y_hat, 'true': y, 'label': label}
    
    def training_epoch_end(self, training_step_outputs):
        # Manually trigger StopIteration due to ligntning module unable to reset DALI pipeline
        try:
            self.train_loader.next()
        except StopIteration:
            pass
    
    def validation_epoch_end(self, validation_step_outputs):
        # Manually trigger StopIteration due to ligntning module unable to reset DALI pipeline
        try:
            self.val_loader.next()
        except StopIteration:
            pass

    def test_epoch_end(self, test_step_outputs):
        # 1. log metrics
        self.logger.log_hyperparams(self.hparams, self.test_hpmetrics0.compute())
        self.logger.log_hyperparams(self.hparams, self.test_hpmetrics1.compute())

        # 2. plot a scatterplot to show correlation between prediction and target
        # collect outputs from each batch
        out_preds = []
        out_trues = []
        out_labels = []
        for outs in test_step_outputs:
            out_preds.append(outs['pred'])
            out_trues.append(outs['true'])
            out_labels.append(outs['label'])
        
        # gather from ddp processes
        out_preds = self.all_gather(torch.stack(out_preds))
        out_trues = self.all_gather(torch.stack(out_trues))
        out_labels = self.all_gather(torch.stack(out_labels))

        # plot figure in the main process
        if self.local_rank == 0: 
            out_preds = out_preds.detach().cpu().numpy().flatten()
            out_trues = out_trues.detach().cpu().numpy().flatten()
            out_labels = out_labels.detach().cpu().numpy().flatten()

            for l in [0, 1]:
                fig = plt.figure(figsize=(12, 12))
                ax = sns.scatterplot(x=out_preds[out_labels==l], y=out_trues[out_labels==l])
                ax.set_xlim(left=0, right=8)
                ax.text(0.1, 0.8, f"pearsonr correlation efficient/p-value \n{pearsonr(out_preds[out_labels==l], out_trues[out_labels==l])}", transform=plt.gca().transAxes)
                self.logger.experiment.add_figure(f"Prediction vs True on test dataset with label {l}", fig)

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        device_id = self.local_rank
        # this is only for Titanv GPU server
        #device_id = 2 if device_id == 1 else device_id
        
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        print(f"local rank {self.local_rank}, device module is on {self.device}, global rank {self.global_rank} in world {num_shards}")

        data_keys = OrderedDict([
            ('seq', True),
            ('chroms', True),
            ('target', True),
            ('label', True)
        ])

        train_pipes = [tfrecord_pipeline(self.train_path, batch_size=int(self.batch_size/num_shards), 
                    device='gpu', num_threads=12, device_id=device_id, shard_id=shard_id, num_shards=num_shards, 
                    random_shuffle=True, reader_name="train", scaler_means=self.scaler_means, scaler_vars=self.scaler_vars, target_vlog=self.target_vlog, **data_keys)]
        val_pipes = [tfrecord_pipeline(self.val_path, batch_size=int(self.batch_size/num_shards), 
                    device='gpu', num_threads=12, device_id=device_id, shard_id=shard_id, num_shards=num_shards, 
                    random_shuffle=True, reader_name="val", scaler_means=self.scaler_means, scaler_vars=self.scaler_vars, target_vlog=self.target_vlog, **data_keys)]
        test_pipes = [tfrecord_pipeline(self.test_path, batch_size=int(self.batch_size/num_shards), 
                    device='gpu', num_threads=12, device_id=device_id, shard_id=shard_id, num_shards=num_shards, 
                    random_shuffle=True, reader_name="test", scaler_means=self.scaler_means, scaler_vars=self.scaler_vars, target_vlog=self.target_vlog, **data_keys)]
        for pipe in train_pipes: pipe.build()
        for pipe in val_pipes: pipe.build()
        for pipe in test_pipes: pipe.build()

        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] for k in self.output_map]
                
        self.train_loader = LightningWrapper(train_pipes, list(data_keys.keys()), reader_name='train', auto_reset=True)
        self.val_loader = LightningWrapper(val_pipes, list(data_keys.keys()), reader_name='val', auto_reset=True)
        self.test_loader = LightningWrapper(test_pipes, list(data_keys.keys()), reader_name='test', auto_reset=True)
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

@pl_cli.MODEL_REGISTRY    
class BichromSeqOnly(BichromDataLoaderHook):
    def __init__(self, data_config, batch_size=512, num_dense=1):
        print(f"BE ADVISED: You are using Seq-Only model")
        super().__init__(data_config, batch_size)
        self.num_dense = num_dense
        self.save_hyperparameters()
        
        self.model_foot = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4, 256, 24)),
            ('relu1', nn.ReLU()),
            ('batchnorm1', nn.BatchNorm1d(256)),
            ('maxpooling1', nn.MaxPool1d(15, 15)),
            ('permute2', permute()),
            ('lstm', nn.LSTM(256, 32, batch_first=True)),
            ('extrat_tensor', extract_tensor()),
            ('dense_aug', nn.Linear(32, 512))
            ]))
        self.model_dense_repeat = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.model_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU()
        )

    def forward(self, seq, chroms):
        y_hat = self.model_foot(seq)
        for i in range(self.num_dense):
            y_hat = self.model_dense_repeat(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

@pl_cli.MODEL_REGISTRY
class Bichrom(BichromDataLoaderHook):
    """
    Early integration of sequence and chromatin info
    """
    def __init__(self, data_config, batch_size=512, chroms_channel=12, conv1d_filter=256, lstm_out=32, dense_aug_feature=512, num_dense=1, seqonly=False):
        print(f"BE ADVISED: You are using Bichrom model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(data_config, batch_size)
        self.num_dense = num_dense
        self.conv1d_filter = conv1d_filter
        self.lstm_out = lstm_out
        self.dense_aug_feature = dense_aug_feature
        self.chroms_channel = chroms_channel
        self.seqonly = seqonly
        self.save_hyperparameters()

        if self.seqonly:
            self.model_foot = nn.Conv1d(4, self.conv1d_filter, 25, bias=False)
        else:
            self.model_foot = nn.Conv1d(4 + self.chroms_channel, self.conv1d_filter, 25, bias=False)
        self.model_body = nn.Sequential(OrderedDict([
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter)),
            ('maxpooling1', nn.MaxPool1d(15, 15)),
            ('permute2', permute()),
            ('lstm', nn.LSTM(self.conv1d_filter, self.lstm_out, batch_first=True)),
            ('extrat_tensor', extract_tensor()),
            ('dense_aug', nn.Linear(self.lstm_out, self.dense_aug_feature)),
            ('relu_aug', nn.LeakyReLU())
            ]))
        dense_repeat_dict = OrderedDict([])
        for i in range(1, self.num_dense+1):
            dense_repeat_dict[f"dense_repeat_{i}"] = nn.Sequential(OrderedDict([
                                                    (f'linear_repeat_{i}', nn.Linear(self.dense_aug_feature, self.dense_aug_feature)),
                                                    (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                    (f'dropout_repeat_{i}', nn.Dropout(0.5))
                                                    ]))
        self.model_dense_repeat = nn.Sequential(dense_repeat_dict)
        self.model_head = nn.Sequential(OrderedDict([
            ('linear_head', nn.Linear(512, 1)),
            ('relu_head', nn.LeakyReLU())
            ]))

    def forward(self, seq, chroms):
        if self.seqonly:
            y_hat = seq
        else:
            y_hat = torch.cat([seq, chroms], dim=1)
        y_hat = self.model_foot(y_hat)
        y_hat = self.model_body(y_hat)
        y_hat = self.model_dense_repeat(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

class bpnet_dilation(nn.Module):
    def __init__(self, channel=64, i=1):
        super().__init__()
        self.channel = channel
        self.i = i
        
        self.conv = nn.Conv1d(self.channel, self.channel, 3, padding=2**self.i, dilation=2**self.i)
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        conv_x = self.conv(x)
        conv_x = self.relu(conv_x)
        return torch.add(x, conv_x)

@pl_cli.MODEL_REGISTRY
class BichromConvDilated(BichromDataLoaderHook):
    """
    Use dilated convolutional layer instead of LSTM in model body
    This one follows the BPnet design style, which means dilation_rate increase exponentially by layer
    """
    def __init__(self, data_config, batch_size=512, chroms_channel=12, conv1d_filter=256, num_dilated=9, seqonly=False):
        print(f"BE ADVISED: You are using Dilated model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(data_config, batch_size)
        self.conv1d_filter = conv1d_filter
        self.num_dilated = num_dilated
        self.chroms_channel = chroms_channel
        self.seqonly = seqonly
        self.save_hyperparameters()

        if seqonly:
            self.model_foot = nn.Sequential(OrderedDict([
                ('conv_chrom1', nn.Conv1d(4, self.conv1d_filter, 25, bias=False)),
                ('relu1', nn.LeakyReLU()),
                ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter))
                ]))
        else:
            self.model_foot = nn.Sequential(OrderedDict([
                ('conv_chrom1', nn.Conv1d(4 + self.chroms_channel, self.conv1d_filter, 25, bias=False)),
                ('relu1', nn.LeakyReLU()),
                ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter))
                ]))
        dilated_dict = OrderedDict([])
        for i in range(1, self.num_dilated+1):
            dilated_dict[f'conv_dilated_{i}'] = bpnet_dilation(self.conv1d_filter, i)
        self.model_dilated_repeat = nn.Sequential(dilated_dict)
        self.model_head = nn.Sequential(OrderedDict([
            ('globalAvgPool1D', nn.AvgPool1d(476)),
            ('squeeze', squeeze()),
            ('linear_head', nn.Linear(self.conv1d_filter, 1)),
            ('relu_head', nn.LeakyReLU())
            ]))
    
    def forward(self, seq, chroms):
        if self.seqonly:
            y_hat = seq
        else:
            y_hat = torch.cat([seq, chroms], dim=1)
        y_hat = self.model_foot(y_hat)
        y_hat = self.model_dilated_repeat(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

@pl_cli.MODEL_REGISTRY
class BichromConvFullDilated(BichromDataLoaderHook):
    """
    Use dilated convolutional layer instead of LSTM in model body
    This one follows the BPnet design style, which means dilation_rate increase exponentially by layer
    """
    def __init__(self, data_config, batch_size=512, chroms_channel=12, conv1d_filter=256, num_dilated=9, seqonly=False):
        print(f"BE ADVISED: You are using Full Receptive field Dilated model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(data_config, batch_size)
        self.conv1d_filter = conv1d_filter
        self.num_dilated = num_dilated
        self.chroms_channel = chroms_channel
        self.seqonly = seqonly
        self.save_hyperparameters()

        self.model_foot = nn.Sequential(OrderedDict([
            ('conv_chrom1', nn.Conv1d(4 if seqonly else (4 + self.chroms_channel), self.conv1d_filter, 25, bias=False)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter))
            ]))
        dilated_dict = OrderedDict([])
        for i in range(0, self.num_dilated):
            dilated_dict[f'conv_dilated_{i}'] = bpnet_dilation(self.conv1d_filter, i)
        self.model_dilated_repeat = nn.Sequential(dilated_dict)
        self.model_head = nn.Sequential(OrderedDict([
            ('globalAvgPool1D', nn.AvgPool1d(476)),
            ('squeeze', squeeze()),
            ('linear_head', nn.Linear(self.conv1d_filter, 1)),
            ('relu_head', nn.LeakyReLU())
            ]))
    
    def forward(self, seq, chroms):
        if self.seqonly:
            y_hat = seq
        else:
            y_hat = torch.cat([seq, chroms], dim=1)
        y_hat = self.model_foot(y_hat)
        y_hat = self.model_dilated_repeat(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

@pl_cli.MODEL_REGISTRY
class BichromNoLSTM(BichromDataLoaderHook):
    """
    Early integration of sequence and chromatin info
    """
    def __init__(self, data_config, batch_size=512, chroms_channel=12, conv1d_filter=256, dense_aug_feature=512, num_conv=1, num_dense=1, seqonly=False):
        print(f"BE ADVISED: You are using Bichrom model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(data_config, batch_size)
        self.num_conv = num_conv
        self.num_dense = num_dense
        self.conv1d_filter = conv1d_filter
        self.dense_aug_feature = dense_aug_feature
        self.chroms_channel = chroms_channel
        self.seqonly = seqonly
        self.save_hyperparameters()

        self.model_foot = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4 if self.seqonly else 4 + self.chroms_channel, self.conv1d_filter, 25, bias=False)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter))
            ]))
        conv_repeat_dict = OrderedDict([])
        for i in range(1, self.num_conv+1):
            conv_repeat_dict[f"conv_repeat_{i}"] = nn.Sequential(OrderedDict([
                                                                (f'conv1d_repeat_{i}', nn.Conv1d(self.conv1d_filter, self.conv1d_filter, 25, padding=12, bias=False)),
                                                                (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                                (f'batchnorm_repeat_{i}', nn.BatchNorm1d(self.conv1d_filter))
                                                                ]))
        self.model_conv_repeat = nn.Sequential(conv_repeat_dict)
        self.model_body = nn.Sequential(OrderedDict([
            ('globalAvgPool1D', nn.AvgPool1d(476)),
            ('squeeze', squeeze()),
            ('dense_aug', nn.Linear(self.conv1d_filter, self.dense_aug_feature)),
            ('relu1', nn.LeakyReLU()),
            ]))
        dense_repeat_dict = OrderedDict([])
        for i in range(1, self.num_dense+1):
            dense_repeat_dict[f"dense_repeat_{i}"] = nn.Sequential(OrderedDict([
                                                    (f'linear_repeat_{i}', nn.Linear(self.dense_aug_feature, self.dense_aug_feature)),
                                                    (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                    (f'dropout_repeat_{i}', nn.Dropout(0.5))
                                                    ]))
        self.model_dense_repeat = nn.Sequential(dense_repeat_dict)
        self.model_head = nn.Sequential(OrderedDict([
            ('linear_head', nn.Linear(512, 1)),
            ('relu_head', nn.LeakyReLU())
            ]))

    def forward(self, seq, chroms):
        if self.seqonly:
            y_hat = seq
        else:
            y_hat = torch.cat([seq, chroms], dim=1)
        y_hat = self.model_foot(y_hat)
        if self.num_conv > 0: y_hat = self.model_conv_repeat(y_hat)
        y_hat = self.model_body(y_hat)
        if self.num_dense > 0: y_hat = self.model_dense_repeat(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)

def main():
    cli = MyLightningCLI(save_config_overwrite=True)

if __name__ == "__main__":
    main()
