# Bichrom
Bichrom provides a framework for modeling, interpreting, and visualizing the joint sequence and chromatin landscapes that determine TF-DNA binding dynamics.

> WARNING: This branch is for predicting TF binding strength instead of bound/unbound, still under construction...

## Citation
Srivastava, D., Aydin, B., Mazzoni, E.O. et al. An interpretable bimodal neural network characterizes the sequence and preexisting chromatin predictors of induced transcription factor binding. Genome Biol 22, 20 (2021). 
https://doi.org/10.1186/s13059-020-02218-6


## Installation and Requirements 

We suggest using anaconda to create a virtual environment using the provided YAML configuration file:
`conda env create -f bichrom.yml -n <env_name>`  

Enter the created environment, then install NVIDIA DALI by:

`python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110`

## Usage

### Step 1 - Construct Bichrom Input Data

Clone and navigate to the Bichrom repository. 
```
# Activate conda environment 
source activate bichrom

cd construct data
usage: construct_data.py [-h] -info INFO -fa FA -blacklist BLACKLIST -len LEN
                         -acc_domains ACC_DOMAINS -chromtracks CHROMTRACKS
                         [CHROMTRACKS ...] -peaks PEAKS -o OUTDIR

Construct Training Data For Bichrom

optional arguments:
  -h, --help            show this help message and exit
  -info INFO            Genome sizes file
  -fa FA                The fasta file for the genome of interest
  -blacklist BLACKLIST  Blacklist file for the genome of interest
  -len LEN              Size of training, test and validation windows
  -acc_domains ACC_DOMAINS
                        Bed file with accessible domains
  -chromtracks CHROMTRACKS [CHROMTRACKS ...]
                        A list of BigWig files for all input chromatin
                        experiments
  -peaks PEAKS          A ChIP-seq or ChIP-exo peak file in multiGPS file
                        format
  -o OUTDIR, --outdir OUTDIR
                        Output directory for storing train, test data
  -p PROCESSORS         Number of processors
  -val_chroms CHROMOSOME
                        Space-delimited chromosome names would be used as validation dataset
  -test_chroms CHROMOSOME
                        Space-delimited chromosome names would be used as test dataset
```

**Required Arguments**

**info**:   
This is a standard genome sizes file, recording the size of each chromosome. It contains 2 tab-separated columns containing the chromosome name and chromosome size.  
For an example file, please see: `sample_data/mm10.info`.  
Genome sizes files are typically available from the UCSC Genome Browser (https://genome.ucsc.edu)

**fa**:  
This is a fasta file from which train, test and validation data should be constructed. 

**len**:  
Length of training, test and validation windows. (**Recommended=500**)

**acc_domains**:   
A BED file containing accessibility domains in the cell-type of interest. This will be used for sampling regions from accessible chromatin when constructing the Bichrom training data.  
For an example file, please see: `sample_data/mES_atacseq_domains.bed`.

**chromtracks**:   
One or more BigWig files containing histone ChIP-seq or ATAC-seq data. 

**peaks**:  
ChIP-seq or ChIP-exo TF peaks in the multiGPS file format. Each peak is represented as **chromosome:midpoint**.  
For an example file, please see: `sample_data/Ascl1.events`.

**nbins**:  
The number of bins to use for binning the chromatin data. (**Recommended=10-20**. Note that with an increase in resolution and **nbins** (or decrease in bin size), the memory requirements will increase.)

**o**:   
Output directory for storing output train, test and validation datasets. 

**blacklist** (optional):   
A blacklist BED file, with artifactual regions to be excluded from the training.  
For an example file, please see: `sample_data/mm10_blacklist.bed`.

**p** (optional):    
Number of processors, default is 1.    
It is suggested to provide more cores to speed up training sample preparation

#### Step 1 - Output 
construct_data.py will produce train, test and validation datasets in the specified output directory.
This function will also produce a configuration file called **bichrom.yaml**, which can be used as input to run Bichrom. This configuration file stores the paths to the created train, test and validation datasets. 


### Step 2 - Train Model

Model training integrates [Pytorch Lightning CLI interface](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html)

```
To view help:   
python trainNN/train.py -h
usage: train.py [-h] [-c CONFIG]
                [--print_config [={comments,skip_null,skip_default}+]]
                {fit,validate,test,predict,tune} ...

pytorch-lightning trainer command line tool

options:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file in json or
                        yaml format.
  --print_config [={comments,skip_null,skip_default}+]
                        Print configuration and exit.

subcommands:
  For more details of each subcommand add it as argument
  followed by --help.

  {fit,validate,test,predict,tune}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the
                        validation set.
    test                Perform one evaluation epoch over the
                        test set.
    predict             Run inference on your data.
    tune                Runs routines to tune hyperparameters
                        before training.

```
  
**Required Arguments**: 

**--config**

This is not required, but for consistency and reproducibility, it's highly suggested tuning the parameters in `config.yaml` file, then provide it to the training script. Things you might want to modify include:

  - trainer.logger.save_dir/trainer.logger.version: where tensorboard logger, checkpoint and config will be stored
  - trainer.nodes/trainer.gpus/trainer.strategy: adjust these parameters to fit your machine capacity
  - trainer.precision: use 16 if you want to use half-precision training, this is not default cuz half-precision training could fail sometimes
  - optimizer.lr/optimizer.betas/...: Adam optimizer hyperparameters

**--model**

Pick the model you would like to train, currently there are mainly two model architectures available: [Bichrom, BichromConvDilated], hyperparameters that can be tuned for each model are:

    Bichrom: data_config(Required), batch_size=512, conv1d_filter=256, lstm_out=32, num_dense=1, seqonly=false
  
    BichromConvDilated: data_config(Required), batch_size=512, conv1d_filter=256, num_dilated=9, seqonly=false

To specify the model and corresponding hyperparameters, add arguments in the following way:

    python trainNN/train.py fit --model Bichrom --model.data_config bichrom.yml --model.batch_size 256 --model.conv1d_filter=128 --model.seqonly=true
    
**Example**:

```
python trainNN/train.py fit --config config.yaml --model Bichrom --model.batch_size 256 --model.conv1d_filter=128 --model.seqonly=true --data.data_config bichrom_out/test1/step1_output/bichrom.yaml
```

Outputs will be stored under the folder specified by `trainer.logger.save_dir`, which would be `lightning_logs/version_0`(described in `config.yaml`) in this example

### Step 3 - Test Model

**Example**:

Supply the config file and the checkpoint in the output directory of model training to `test` command, as example:

```
python trainNN/train.py test --config lightning_logs/version_1/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch=11-step=48.ckpt
 --trainer.logger TensorBoardLogger --trainer.logger.save_dir test_logs/
```

Outputs will be stored under the folder specified by `trainer.logger.save_dir`, which would be `test_logs/version_0` in this example

### Tensorboard visualization

**Example**:

To visualize the metrics, lauch a tensorboard session on the output directory of train/test step

```
tensorboard serve --logdir /path/to/output/dir/
```

### Prediction

To use the trained model for prediction on new data, supply the config file and checkpoint in the output directory of model training as did for test step, one additional thing to provide is the bed file (`--data.pred_bed`) describing the regions you want to predict on.

Requirements of bed file columns:

- chrom
- start
- end
- name (set as "." if you don't have this info)
- score (set as "." if you don't have this info)
- strand (set as "." if you don't have this info)

**Example**:

use model in the following way:

```
python trainNN/train.py predict --config lightning_logs/version_1/config.yaml --ckpt_path test/lightning_logs/version_0/checkpoints/epoch=11-step=48.ckpt
 --trainer.logger TensorBoardLogger --trainer.logger.save_dir predict_logs/ --data.pred_bed bichrom_out/test1/step1_output/data_test.bed --data.num_workers 16
```

The prediction `model_pred.txt` will be saved under `predict_logs/version_0` directory

> adjust `--data.num_workers` according to your machine capacity (number of processors) to improve performance
