
export DEEPJETCORE=../DeepJetCore

THISDIR=`pwd`
cd $DEEPJETCORE
source gpu_env.sh
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
export DEEPJET=`pwd`

# for caltech cluster:
#rm -rf $HOME/.nv
#export CUDA_CACHE_PATH=/tmp/$USER/cuda/
#mkdir -p $CUDA_CACHE_PATH

# for Maxwell (DESY) cluster
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/TensorFlow/lib64/

# for RWTH cluster
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rwthfs/rz/SW/cuda/cudnn/7.0.5/90/lib64/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local_rwth/sw/cuda/9.0.176/lib64/
#module load cuda/90
#module load cudnn/7.0.5

