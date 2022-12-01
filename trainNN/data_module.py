
description = """
    Lightning Data Module for model training
    Given bed file, return sequence and chromatin info
"""

from math import sqrt, ceil
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import yaml
import pyfasta
import pyBigWig
import torch
from itertools import islice
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import cli as pl_cli

import webdataset as wds

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
        seqMatrix = np.zeros((4, len(dnaSeq)), dtype=np.float32)
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
        self.seq_transform = seq_transform

        self.bigwig_files = config["params"]["chromatin_tracks"]
        self.scaler_mean = config["params"]["scaler_mean"]
        self.scaler_var = config["params"]["scaler_var"]
    
    def initialize(self):
        self.genome_pyfasta = pyfasta.Fasta(self.config["params"]["fasta"])
        #self.tfbam = pysam.AlignmentFile(self.config["train_bichrom"]["tf_bam"])
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
                m = (np.nan_to_num(bigwig.values(entry.chrom, entry.start, entry.end))).astype(np.float32)
                if entry.strand == "-": m = m[::-1] # reverse if needed
                if self.scaler_mean and self.scaler_var:
                    m = (m - self.scaler_mean[idx])/sqrt(self.scaler_var[idx])
                ms.append(m)
        except RuntimeError as e:
            print(e)
            raise Exception(f"Failed to extract chromatin {self.bigwig_files[idx]} information in region {entry}")
        ms = np.vstack(ms)
        ## target: read count in region
        #target = self.tfbam.count(entry.chrom, entry.start, entry.end)

        return seq, ms

    def rev_comp(self, inp_str):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
                   'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
        outp_str = list()
        for nucl in inp_str:
            outp_str.append(rc_dict[nucl])
        return ''.join(outp_str)[::-1] 

def count_lines(fp):
    with open(fp, 'r') as f:
        for count, line in enumerate(f):
            pass
    return count+1

def _split_by_node(src, global_rank, world_size):
    if world_size > 1:
        for s in islice(src, global_rank, None, world_size):
            yield s
    else:
        for s in src:
            yield s

split_by_node = wds.pipelinefilter(_split_by_node)

def _scale_chrom(sample, scaler_mean, scaler_std):
    # standardize chrom by provided mean and std
    seq, chrom, target, label = sample
    
    chrom = np.divide(chrom - scaler_mean, scaler_std, dtype=np.float32)

    return seq, chrom, target, label

scale_chrom = wds.pipelinefilter(_scale_chrom)

def _target_vlog(sample):
    # take log(n+1) on target
    seq, chrom, target, label = sample

    target = np.log(target + 1, dtype=np.float32)

    return seq, chrom, target, label

target_vlog = wds.pipelinefilter(_target_vlog)

class SeqChromDataModule(pl.LightningDataModule):
    def __init__(self, data_config, pred_bed, dataset_train="train_bichrom", num_workers=8, batch_size=512, seed=1):
        super().__init__()
        self.config = yaml.safe_load(open(data_config, 'r'))
        self.pred_bed = pred_bed
        self.dataset_train = dataset_train
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.scaler_mean = np.array(self.config["params"]["scaler_mean"], dtype=float).reshape(-1, 1)
        self.scaler_std = np.sqrt(np.array(self.config["params"]["scaler_var"], dtype=float).reshape(-1, 1))

        # load bed file to infer dataset length
        self.train_dataset_size = count_lines(self.config["train_bichrom"]["bed"]) * 2 # +/- strand
        self.val_dataset_size = count_lines(self.config["val"]["bed"])
        self.test_dataset_size = count_lines(self.config["test"]["bed"])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            global_rank = self.trainer.global_rank
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            global_rank = 0
            world_size = 1

        self.batch_size_per_rank = int(self.batch_size/world_size)

        if stage in ["fit", "validate", "test"] or stage is None:

            self.train_loader = wds.DataPipeline(
                wds.SimpleShardList(self.config[self.dataset_train]["webdataset"]),
                wds.shuffle(100, rng=random.Random(self.seed)),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(1000, rng=random.Random(self.seed)),
                wds.decode(),
                wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
                wds.map(scale_chrom(self.scaler_mean, self.scaler_std)),
                wds.map(target_vlog()),
            ) 

            self.val_loader = wds.DataPipeline(
                wds.SimpleShardList(self.config["val"]["webdataset"]),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode(),
                wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
                wds.map(scale_chrom(self.scaler_mean, self.scaler_std)),
                wds.map(target_vlog()),
            )

            self.test_loader = wds.DataPipeline(
                wds.SimpleShardList(self.config["test"]["webdataset"]),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode(),
                wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
                wds.map(scale_chrom(self.scaler_mean, self.scaler_std)),
                wds.map(target_vlog()),
            )

        if (stage == "predict" or stage is None) and (not self.pred_bed is None):
            self.predict_dataset = SeqChromDataset(self.pred_bed, self.config)
    
    def train_dataloader(self):
        return wds.WebLoader(self.train_loader.repeat(2), num_workers=self.num_workers, batch_size=self.batch_size_per_rank).with_epoch(ceil(self.train_dataset_size/self.batch_size)) # pad the last batch if there is remainder

    def val_dataloader(self):
        return wds.WebLoader(self.val_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)

    def test_dataloader(self):
        return wds.WebLoader(self.test_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size_per_rank, num_workers=self.num_workers, worker_init_fn=worker_init_fn, prefetch_factor=4)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize()