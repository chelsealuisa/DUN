#!/bin/bash

set -e

if [ -z "$1" ]
then
    echo "Error: no dataset type given"
    exit 1
fi
if [ -z "$2" ]
then
    echo "Error: no epochs given"
    exit 1
fi

group=$1
epochs=$2

script="train_${group}_alb_slurm.py"

# define dicitonaries
declare -A layers
layers=(["DUN"]="10" ["Dropout"]="3" ["MFVI"]="3" ["SGD"]="3")
declare -A init_train
init_train=(["wiggle"]="10" ["agw_1d"]="10" ["andrew_1d"]="10" ["matern_1d"]="10" ["my_1d"]="10" ["boston"]="20" ["concrete"]="50" ["energy"]="50" ["kin8nm"]="50" ["naval"]="50" ["power"]="50" ["protein"]="50" ["wine"]="50" ["yacht"]="20")
declare -A query_size
query_size=(["wiggle"]="10" ["agw_1d"]="10" ["andrew_1d"]="5" ["matern_1d"]="10" ["my_1d"]="10" ["boston"]="20" ["concrete"]="20" ["energy"]="20" ["kin8nm"]="20" ["naval"]="20" ["power"]="20" ["protein"]="20" ["wine"]="20" ["yacht"]="10")
declare -A n_queries
n_queries=(["wiggle"]="20" ["agw_1d"]="30" ["andrew_1d"]="14" ["matern_1d"]="30" ["my_1d"]="30" ["boston"]="17" ["concrete"]="30" ["energy"]="30" ["kin8nm"]="30" ["naval"]="30" ["power"]="30" ["protein"]="30" ["wine"]="30" ["yacht"]="20")
declare -A output_folders
output_folders=(["toy"]="saves" ["reg"]="saves_regression" ["class"]="saves_classification" ["img"]="saves_images")

#methods=("DUN" "Dropout" "MFVI" "SGD")
methods=("DUN")
datasets=("boston" "concrete" "energy" "kin8nm" "power" "protein" "naval" "wine" "yacht")
inference="DUN"

# Create a fresh file
> alb_jobs.txt

# Loop through run_id, training method, acquisition strategy
for dataset in ${datasets[@]}
do
    for rep in `seq 0 39`
    do
        echo "${script} --output_folder ${output_folders[${group}]} --inference ${inference} --N_layers ${layers[$inference]} --overcount 1 --width 100 --n_epochs ${epochs} --dataset ${dataset} --lr 0.001 --wd 0.0001 --clip_var --sampling --T 10000 --run_id ${rep}" >> alb_jobs.txt
    done
done
