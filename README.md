

DeepJet: Repository for training and evaluation of deep neural networks for Jet identification
===============================================================================

This package depends on DeepJetCore - original at (https://github.com/DL4Jets/DeepJetCore)
For work on DeepDoubleX (running on DESY/Maxwell GPU cluster or caltech cluster) follow instructions at:
https://github.com/DeepDoubleB/DeepJetCore

Setup
==========

The DeepJet package and DeepJetCore have to share the same parent directory

Usage (Maxwell)
==============

Connect to desy and maxwell
```
ssh max-wgs
```

To run interactively:
```
salloc -N 1 --partition=all --constraint=GPU --time=<minutes>
```

Alternatively add the following to your .bashrc to get allocation for x hours with ``` getgpu x ```
```
getgpu () {
   salloc -N 1 --partition=all --constraint=GPU --time=$((60 * $1))
}
```

ssh to the machine that was allocated to you. For example
```
ssh max-wng001
```
```
cd <your working dir>/DeepJet
source gpu_env.sh
```
(Edit gpu_env.sh to uncomment the line corresponding to the cluster in use.)

The preparation for the training consists of the following steps
====

- define the data structure for the training (example in modules/datastructures/TrainData_DeepDoubleX.py)
- convert the root file to the data strucure for training using DeepJetCore tools:

- You can use the following script to create the lists (if you store the files in a train and test directory within one parent you can only specify test
```
  python list_writer.py --train <path/to/directory/of/files/train> --test <path/to/directory/of/files/test>  
  # when not specified otherwise test_list.txt is searched for in "path_to_train_files".replace('train','test')
``` 
```
  INDIR=/needs/some/disk/space
  mkdir INDIR
  convertFromRoot.py -i train_list.txt -o $INDIR/dctrain -c TrainData_DeepDoubleX_reference
```
Training
====

Run the training
```
  python Train/Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training  --batch 4096 --epochs 10

```
Evaluation
====

After the training has finished, the performance can be evaluated.
The evaluation consists of a few steps:

1) converting the test data
```
  convertFromRoot.py -i test_list.txt -o $INDIR/dctest --testdatafor $INDIR/training/trainsamples.dc
```

2.a) Evaluate to get a pandas df and automatic plots

```
  python Train/Eval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/eval
```

Output .pkl file and some plots will be stored in $INDIR/eval

2.b) Evaluate to get a tree friend for input test files.

```
predict.py $INDIR/KERAS_model.h5  $INDIR/dctest/dataCollection.dc $INDIR/output
```
This creates output trees. and a tree_association.txt file that is input to the plotting tools

There is a set of plotting tools with examples in
DeepJet/Train/Plotting


To use Maxwell Batch (SLURM)
====
Example config file can be found in run/
```
# To run binary Hcc vs QCD training
sbatch run/baseDDC.sh

# To run binary Hcc vs Hbb training
sbatch run/baseDDCvB.sh

# To run multiclassifier for Hcc, Hbb, QCD (gcc, gbb, Light)
sbatch run/baseDDX.sh

# To see job output updated in real time
tail -f run/run-<jobid>.out 
# To show que
squeue -u username 
# To cancel a job 
scancel jobid # To cancel job
scancel -u username
```


