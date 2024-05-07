#!/bin/bash

# bash subprocess_train.sh ml_data/CCLE-CCLE/split_0 ml_data/CCLE-CCLE/split_0 out_model/CCLE/split_0
# CUDA_VISIBLE_DEVICES=5 bash subprocess_train.sh ml_data/CCLE-CCLE/split_0 ml_data/CCLE-CCLE/split_0 out_model/CCLE/split_0

# Need to comment this when using ' eval "$(conda shell.bash hook)" '
# set -e

# Activate conda env for model using "conda activate myenv"
# https://saturncloud.io/blog/activating-conda-environments-from-scripts-a-guide-for-data-scientists
# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# This doesn't work w/o eval "$(conda shell.bash hook)"

#echo "Allow conda commands in shell script by running 'conda shell.bash hook'"
#eval "$(conda shell.bash hook)"
#echo "Activated conda commands in shell script"
#conda activate $CONDA_ENV
#source activate $CONDA_ENV
#conda_path=${dirname `which conda`}
#source $conda_path/activate $CONDA_ENV
#source activate $CONDA_ENV
#echo "Activated conda env $CONDA_ENV"



train_ml_data_dir=$1
val_ml_data_dir=$2
model_outdir=$3
epochs=$4
batch_size=$5
learning_rate=$6
direct_gene_weight_param=$7
num_hiddens_genotype=$8
num_hiddens_final=$9
inter_loss_penalty=${10}
eps_adam=${11}
beta_kl=${12}
CUDA_VISIBLE_DEVICES=${13}
echo "train_ml_data_dir: $train_ml_data_dir"
echo "val_ml_data_dir:   $val_ml_data_dir"
echo "model_outdir:      $model_outdir"
echo "epochs:    $epochs"
echo "batch_size:    $batch_size"
echo "learning_rate:   $learning_rate"
echo "direct_gene_weight_param:  $direct_gene_weight_param"
echo "num_hiddens_genotype:  $num_hiddens_genotype"
echo "num_hiddens_final:  $num_hiddens_final"
echo "inter_loss_penalty:  $inter_loss_penalty"
echo "eps_adam:   $eps_adam"
echo "beta_kl:   $beta_kl"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

### Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

echo ${IMPROVE_MODEL_DIR}


if [ -d ${IMPROVE_DATA_DIR} ]; then
    if [ "$(ls -A ${IMPROVE_DATA_DIR})" ] ; then
        echo "using data from ${IMPORVE_DATA_DIR}"
    else
        ./candle_glue.sh
        echo "using original data placed in ${IMPROVE_DATA_DIR}"
    fi
fi

export IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR}

# epochs=10
epochs=10
# epochs=50

image_file='/homes/ac.rgnanaolivu/improve_data_dir/DrugCell/images/DrugCell_tianshu:0.0.1-20240429.sif'



cmd="singularity exec --nv --bind ${IMPROVE_MODEL_DIR} ${image_file} python DrugCell_train_improve.py --train_ml_data_dir=$train_ml_data_dir --val_ml_data_dir=$val_ml_data_dir --model_outdir=$model_outdir --epochs=$epochs --batch_size=$batch_size --learning_rate=$learning_rate --direct_gene_weight_param=$direct_gene_weight_param --num_hiddens_genotype=$num_hiddens_genotype --num_hiddens_final=$num_hiddens_final --inter_loss_penalty=$inter_loss_penalty --eps_adam=$eps_adam --beta_kl=$beta_kl --cuda_name=$CUDA_VISIBLE_DEVICES"
eval ${cmd}


