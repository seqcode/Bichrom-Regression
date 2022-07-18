
description = """
    Lightning Data Module for model training
    Given bed file, return sequence and chromatin info
"""

from math import sqrt
from collections import OrderedDict

import numpy as np
import pandas as pd

import yaml
import pyfasta
import pyBigWig
import pysam

import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import cli as pl_cli

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from dali_input import tfrecord_pipeline

class DNA2OneHot(object):
    def __init__(self):
        self.DNA2Index = {
            "A": 0,
            "T": 1,
            "G": 2,
            "C": 3 
        }
    
    def __call__(self, dnaSeq):
        seqLen = len(dnaSeq)
        # initialize the matrix as 4 x len(dnaSeq)
        seqMatrix = np.zeros((4, len(dnaSeq)), dtype=int)
        # change the value to matrix
        dnaSeq = dnaSeq.upper()
        for j in range(0, seqLen):
            if dnaSeq[j] == "N": continue
            try:
                seqMatrix[self.DNA2Index[dnaSeq[j]], j] = 1
            except KeyError as e:
                print(f"Keyerror happened at position {j}: {dnaSeq[j]}")
                continue
        return seqMatrix

class SeqChromDataset(Dataset):
    def __init__(self, bed, config=None, seq_transform=DNA2OneHot()):
        self.bed = pd.read_table(bed, header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand' ])

        self.config = config
        self.nbins = config["train_bichrom"]["nbins"]
        self.seq_transform = seq_transform

        self.bigwig_files = config["train_bichrom"]["chromatin_tracks"]
        self.scaler_mean = config["train_bichrom"]["scaler_mean"]
        self.scaler_var = config["train_bichrom"]["scaler_var"]
    
    def initialize(self):
        self.genome_pyfasta = pyfasta.Fasta(self.config["train_bichrom"]["fasta"])
        self.tfbam = pysam.AlignmentFile(self.config["train_bichrom"]["tf_bam"])
        self.bigwigs = [pyBigWig.open(bw) for bw in self.bigwig_files]
    
    def __len__(self):
        return len(self.bed)

    def __getitem__(self, idx):
        entry = self.bed.iloc[idx,]
        # get info in the each entry region
        ## sequence
        sequence = self.genome_pyfasta[entry.chrom][int(entry.start):int(entry.end)]
        sequence = self.rev_comp(sequence) if entry.strand=="-" else sequence
        seq = self.seq_transform(sequence)
        ## chromatin
        ms = []
        try:
            for idx, bigwig in enumerate(self.bigwigs):
                m = (np.nan_to_num(bigwig.values(entry.chrom, entry.start, entry.end)).reshape((self.nbins, -1)).mean(axis=1, dtype=np.float32))
                if entry.strand == "-": m = m[::-1] # reverse if needed
                if self.scaler_mean and self.scaler_var:
                    m = (m - self.scaler_mean[idx])/sqrt(self.scaler_var[idx])
                ms.append(m)
        except RuntimeError as e:
            print(e)
            raise Exception(f"Failed to extract chromatin {self.bigwig_files[idx]} information in region {entry}")
        ms = np.vstack(ms)
        ## target: read count in region
        target = self.tfbam.count(entry.chrom, entry.start, entry.end)

        return seq, ms, target

    def rev_comp(self, inp_str):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
                   'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
        outp_str = list()
        for nucl in inp_str:
            outp_str.append(rc_dict[nucl])
        return ''.join(outp_str)[::-1] 

@pl_cli.DATAMODULE_REGISTRY
class SeqChromDataModule(pl.LightningDataModule):
    def __init__(self, data_config, pred_bed, num_workers=1, batch_size=512, target_vlog=True):
        super().__init__()
        self.config = yaml.safe_load(open(data_config, 'r'))
        self.pred_bed = pred_bed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.target_vlog = target_vlog
        self.scaler_mean = self.config["train_bichrom"]["scaler_mean"]
        self.scaler_var = self.config["train_bichrom"]["scaler_var"]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in ["fit", "validate", "test", "None"]:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            shard_id = self.trainer.global_rank
            num_shards = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {num_shards}")

            data_keys = OrderedDict([
                ('seq', True),
                ('chroms', True),
                ('target', True),
                ('label', True)
            ])

            train_pipes = [tfrecord_pipeline(self.config["train_bichrom"], batch_size=int(self.batch_size/num_shards), 
                        device='gpu', num_threads=12, device_id=device_id, shard_id=shard_id, num_shards=num_shards, 
                        random_shuffle=True, reader_name="train", scaler_means=self.scaler_mean, scaler_vars=self.scaler_var, target_vlog=self.target_vlog, **data_keys)]
            val_pipes = [tfrecord_pipeline(self.config["val"], batch_size=int(self.batch_size/num_shards), 
                        device='gpu', num_threads=12, device_id=device_id, shard_id=shard_id, num_shards=num_shards, 
                        random_shuffle=True, reader_name="val", scaler_means=self.scaler_mean, scaler_vars=self.scaler_var, target_vlog=self.target_vlog, **data_keys)]
            test_pipes = [tfrecord_pipeline(self.config["test"], batch_size=int(self.batch_size/num_shards), 
                        device='gpu', num_threads=12, device_id=device_id, shard_id=shard_id, num_shards=num_shards, 
                        random_shuffle=True, reader_name="test", scaler_means=self.scaler_mean, scaler_vars=self.scaler_var, target_vlog=self.target_vlog, **data_keys)]
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

        if stage == "predict" or stage is None:
            self.predict_dataset = SeqChromDataset(self.pred_bed, self.config)
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize()