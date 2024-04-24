#!/bin/bash


CONDA_ENV='drugcell_python'
CONDA='/homes/ac.rgnanaolivu/miniconda3/condabin/conda'
echo "Allow conda commands in shell script by running 'conda shell.bash hook'"
eval "$(conda shell.bash hook)"
#${CONDA} shell.bash hook
echo "Activated conda commands in shell script"
${CONDA} activate $CONDA_ENV
echo "Activated conda env $CONDA_ENV"
export TF_CPP_MIN_LOG_LEVEL=3

train_ml_data_dir=$1
val_ml_data_dir=$2
model_outdir=$3
echo "train_ml_data_dir: $train_ml_data_dir"
echo "val_ml_data_dir:   $val_ml_data_dir"
echo "model_outdir:      $model_outdir"


# activate conda env for model
#epochs=20

python DrugCell_train_improve.py \
  --train_ml_data_dir=$train_ml_data_dir \
  --val_ml_data_dir=$val_ml_data_dir \
  --model_outdir=$model_outdir 
#  --epochs=$epochs \
#  --learning_rate=$learning_rate \
#  --beta_kl=$beta_kl \
#  --num_hiddens_final=$num_hiddens_final


conda deactivate
echo "Deactivated conda env $CONDA_ENV"
