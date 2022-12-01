
import os, sys
import yaml
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection, MeanSquaredError, PearsonCorrCoef
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns

from data_module import SeqChromDataModule

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
    def __init__(self, chroms_channel):
        super().__init__()
        self.chroms_channel = chroms_channel

        self.metrics = MetricCollection([MeanSquaredError(), PearsonCorrCoef()])
        self.test_hpmetricsall = self.metrics.clone(prefix="hp/test_all_")
        self.test_hpmetrics0 = self.metrics.clone(prefix="hp/test_label0_")
        self.test_hpmetrics1 = self.metrics.clone(prefix="hp/test_label1_")

        self.example_input_array = [torch.zeros(512, 4, 500).index_fill_(1, torch.tensor(2), 1), torch.ones(512, self.chroms_channel, 500)]

    def vlog(self, tensor):
        """
        log(tensor+1) operation
        """
        return torch.log(torch.add(tensor, 1))

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_test_start(self):
        self.logger.log_hyperparams(self.hparams, {i:0 for i in list(self.test_hpmetricsall.keys()) + list(self.test_hpmetrics0.keys()) + list(self.test_hpmetrics1.keys())})

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
        self.log('val_loss', val_loss, sync_dist=True)
        return {'val_loss': val_loss, 'pred': y_hat, 'true': y}

    def test_step(self, batch, batch_idx):
        # define test
        seq, chroms, y, label = batch
        y_hat = self(seq, chroms)

        # compute prediction and loss
        test_loss = F.mse_loss(y_hat, y)

        # compute metrics
        self.test_hpmetricsall(y_hat, y)
        if (label==0).sum()>1: self.test_hpmetrics0(y_hat[label==0], y[label==0])
        if (label==1).sum()>1: self.test_hpmetrics1(y_hat[label==1], y[label==1])

        self.log('test_loss', test_loss, sync_dist=True)
        return {'test_loss': test_loss, 'pred': y_hat, 'true': y, 'label': label}
    
    def predict_step(self, batch, batch_idx):
        seq, chroms = batch
        y_hat = self(seq, chroms)

        return y_hat
    
    def test_epoch_end(self, test_step_outputs):
        # 1. log metrics
        self.logger.log_hyperparams(self.hparams, self.test_hpmetricsall.compute())
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
        
        out_preds = torch.cat(out_preds, 0)
        out_trues = torch.cat(out_trues, 0)
        out_labels = torch.cat(out_labels, 0)

        # plot figure in the main process
        if self.local_rank == 0: 
            out_preds = out_preds.detach().cpu().numpy().flatten()
            out_trues = out_trues.detach().cpu().numpy().flatten()
            out_labels = out_labels.detach().cpu().numpy().flatten()

            df = pd.DataFrame({'predictions': out_preds,
                                'true_target': out_trues,
                                'label': out_labels})
            df.to_csv(os.path.join(self.logger.log_dir, "model_preds_vs_true.txt"), header=True, index=False, sep="\t")

            ## scatterplot on all test data
            fig = plt.figure(figsize=(12, 12))
            ax = sns.scatterplot(x=out_preds, y=out_trues)
            ax.set_xlim(left=0, right=12)
            ax.set_ylim(bottom=0, top=12)
            ax.set_xlabel("Predictions by Model")
            ax.set_ylabel("True target")
            ax.text(0.1, 0.8, f"pearsonr correlation efficient/p-value \n{pearsonr(out_preds, out_trues)}", transform=plt.gca().transAxes)
            self.logger.experiment.add_figure(f"Prediction vs True on whole test dataset", fig)

            ## scatterplot on labeled test data
            for l in [0, 1]:
                fig = plt.figure(figsize=(12, 12))
                ax = sns.scatterplot(x=out_preds[out_labels==l], y=out_trues[out_labels==l])
                ax.set_xlim(left=0, right=12)
                ax.set_ylim(bottom=0, top=12)
                ax.set_xlabel("Predictions by Model")
                ax.set_ylabel("True target")
                ax.text(0.1, 0.8, f"pearsonr correlation efficient/p-value \n{pearsonr(out_preds[out_labels==l], out_trues[out_labels==l])}", transform=plt.gca().transAxes)
                self.logger.experiment.add_figure(f"Prediction vs True on test dataset with label {l}", fig)

    def on_train_epoch_end(self):
        # logging histograms
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_test_epoch_start(self):
        # ensure world size is 1
        if self.trainer.world_size != 1:
            print(f"World size is {self.trainer.world_size}")
            print(f"Please set # of devices as 1, distributed strategy on multiple devices could lead to incorrect prediction tensor shape")
            sys.exit(1)

    def on_predict_epoch_end(self, predict_step_outputs):
        # collect outputs from each batch
        out_preds = []
        for outs in predict_step_outputs[0]: 
            out_preds.append(outs)
        
        # gather from ddp processes
        out_preds = self.all_gather(torch.cat(out_preds))

        # save
        if self.global_rank == 0:
            np.savetxt(os.path.join(self.logger.log_dir, "model_preds.txt"), out_preds.cpu().numpy().flatten(), fmt="%.6f")
        print(f"Saved model predictions into model_preds.txt")
    
class Bichrom(BichromDataLoaderHook):
    """
    Early integration of sequence and chromatin info
    """
    def __init__(self, chroms_channel=12, conv1d_filter=256, lstm_out=32, dense_aug_feature=512, num_dense=1, seqonly=False):
        print(f"BE ADVISED: You are using Bichrom model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(chroms_channel)
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

class BichromConvDilated(BichromDataLoaderHook):
    """
    Use dilated convolutional layer instead of LSTM in model body
    This one follows the BPnet design style, which means dilation_rate increase exponentially by layer
    """
    def __init__(self, chroms_channel=12, conv1d_filter=256, num_dilated=9, seqonly=False):
        print(f"BE ADVISED: You are using Dilated model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(chroms_channel)
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

class BichromConvFullDilated(BichromDataLoaderHook):
    """
    Use dilated convolutional layer instead of LSTM in model body
    This one follows the BPnet design style, which means dilation_rate increase exponentially by layer
    """
    def __init__(self, chroms_channel=12, conv1d_filter=256, num_dilated=9, seqonly=False):
        print(f"BE ADVISED: You are using Full Receptive field Dilated model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(chroms_channel)
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

class BichromNoLSTM(BichromDataLoaderHook):
    """
    Early integration of sequence and chromatin info
    """
    def __init__(self, chroms_channel=12, conv1d_filter=256, dense_aug_feature=512, num_conv=1, num_dense=1, seqonly=False):
        print(f"BE ADVISED: You are using Bichrom model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(chroms_channel)
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

class ConvRepeat(BichromDataLoaderHook):
    """
    Early integration of sequence and chromatin info
    """
    def __init__(self, chroms_channel=12, conv1d_filter=256, dense_aug_feature=512, num_conv=4, num_dense=3, seqonly=False):
        print(f"BE ADVISED: You are using Bichrom model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(chroms_channel)
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
            ('batchnorm', nn.BatchNorm1d(self.dense_aug_feature))
            ]))
        dense_repeat_dict = OrderedDict([])
        for i in range(1, self.num_dense+1):
            dense_repeat_dict[f"dense_repeat_{i}"] = nn.Sequential(OrderedDict([
                                                    (f'linear_repeat_{i}', nn.Linear(self.dense_aug_feature, self.dense_aug_feature)),
                                                    (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                    (f'batchnorm_repeat_{i}', nn.BatchNorm1d(self.dense_aug_feature)),
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

class ConvRepeatVGGLike(ConvRepeat):
    def __init__(self, chroms_channel=12, conv1d_filter=64, conv1d_filter_width=3,dense_aug_feature=512, num_dense=3, seqonly=False):
        super().__init__(chroms_channel, conv1d_filter, dense_aug_feature, 3, num_dense, seqonly)
        self.conv1d_filter_width = conv1d_filter_width

        self.model_foot = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(4 if self.seqonly else 4 + self.chroms_channel, self.conv1d_filter, 25, padding=12, bias=False)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter))
            ]))

        conv_repeat_dict = OrderedDict([])
        for i in range(1, 5):
            conv_repeat_dict[f"conv_repeat_p{i}_reduce"] = nn.Sequential(OrderedDict([
                                                                    (f'conv1d_repeat_{i}', nn.Conv1d(self.conv1d_filter*i, self.conv1d_filter*(i+1), self.conv1d_filter_width, stride=2, padding=int((self.conv1d_filter_width-1)/2), bias=False)),
                                                                    (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                                    (f'batchnorm_repeat_{i}', nn.BatchNorm1d(self.conv1d_filter*(i+1)))
                                                                    ]))
        self.model_conv_repeat = nn.Sequential(conv_repeat_dict)
        self.model_body = nn.Sequential(OrderedDict([
            ('conv1d_reduce', nn.Conv1d(self.conv1d_filter*5, 1, 1, bias=False)),
            ('squeeze', squeeze()),
            ('dense_aug', nn.Linear(32, self.dense_aug_feature)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm', nn.BatchNorm1d(self.dense_aug_feature))
            ]))
        self.model_head = nn.Sequential(OrderedDict([
            ('linear_head', nn.Linear(self.dense_aug_feature, 1)),
            ('relu_head', nn.LeakyReLU())
            ]))
            
class LSTMRepeat(BichromDataLoaderHook):
    """
    Early integration of sequence and chromatin info
    """
    def __init__(self, chroms_channel=12, conv1d_filter=256, num_lstm=1, lstm_out=32, dense_aug_feature=512, num_dense=1, seqonly=False):
        print(f"BE ADVISED: You are using Bichrom model in {'Seq-only' if seqonly else 'Seq + Chrom'} mode...")
        super().__init__(chroms_channel)
        self.num_dense = num_dense
        self.conv1d_filter = conv1d_filter
        self.num_lstm = num_lstm
        self.lstm_out = lstm_out
        self.dense_aug_feature = dense_aug_feature
        self.chroms_channel = chroms_channel
        self.seqonly = seqonly
        self.save_hyperparameters()

        self.model_foot = nn.Sequential(OrderedDict([
            ('conv_chrom1', nn.Conv1d(4 if seqonly else (4 + self.chroms_channel), self.conv1d_filter, 25, bias=False)),
            ('relu1', nn.LeakyReLU()),
            ('batchnorm1', nn.BatchNorm1d(self.conv1d_filter))
            ]))
        self.model_body = nn.Sequential(OrderedDict([
            ('permute0', permute()),
            ('lstm', nn.LSTM(self.conv1d_filter, self.lstm_out, num_layers=self.num_lstm, batch_first=True)),
            ('extrat_tensor0', extract_tensor()),
            ('dense_aug', nn.Linear(self.lstm_out, self.dense_aug_feature)),
            ('relu_aug', nn.LeakyReLU()),
            ('batchnorm_aug', nn.BatchNorm1d(self.dense_aug_feature))
            ]))
        dense_repeat_dict = OrderedDict([])
        for i in range(1, self.num_dense+1):
            dense_repeat_dict[f"dense_repeat_{i}"] = nn.Sequential(OrderedDict([
                                                    (f'linear_repeat_{i}', nn.Linear(self.dense_aug_feature, self.dense_aug_feature)),
                                                    (f'relu_repeat_{i}', nn.LeakyReLU()),
                                                    (f'batchnorm_repeat_{i}', nn.BatchNorm1d(self.dense_aug_feature)),
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
        if self.num_dense > 0: y_hat = self.model_dense_repeat(y_hat)
        y_hat = self.model_head(y_hat)

        return y_hat

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

def main():
    cli = MyLightningCLI(datamodule_class=SeqChromDataModule,
                        seed_everything_default=32,
                        save_config_overwrite=True,
                        trainer_defaults={
                            "callbacks": [
                                ModelCheckpoint(filename="checkpoint_{epoch}-{val_loss:.6f}", 
                                                monitor='val_loss', 
                                                save_last=True, 
                                                save_top_k=1, 
                                                mode='min', 
                                                every_n_epochs=1),
                                EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20),
                                ModelSummary(max_depth=-1)]
                        })

if __name__ == "__main__":
    main()
