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

script="train_${group}_slurm_2kep.py"

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

methods=("DUN" "Dropout" "MFVI")
datasets=("wiggle" "andrew_1d", "agw_1d", "matern_1d", "my_1d")


# Loop through run_id, training method, acquisition strategy
# Create a fresh file
> toy_2kep_jobs.txt
for dataset in ${datasets[@]}
do
    for method in ${methods[@]}
    do
        for rep in `seq 0 39`
        do
            echo "${script} --output_folder ${output_folders[${group}]} --inference ${method} --N_layers ${layers[$method]} --overcount 1 --width 100 --n_epochs ${epochs} --dataset ${dataset} --lr 0.001 --wd 0.0001 --n_queries ${n_queries[$dataset]} --query_size ${query_size[$dataset]} --init_train ${init_train[$dataset]} --query_strategy variance --clip_var --run_id ${rep}" >> toy_2kep_jobs.txt
        done
    done
done
