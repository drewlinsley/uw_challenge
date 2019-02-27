#!/bin/bash

# # Inputs:
# 0. GPU ID
# 1. CSV with model results
# 2. Test experiment name
# 3. Output name

GPU="1"
CKPTCSV="analysis_data/checkpoints_nist_3_ix2v2_25k.csv"
MODELCSV="analysis_data/models_nist_3_ix2v2_25k.csv"
declare -a EXPERIMENTS=("pathfinder_14_test" "nist_3_ix2v2_test")
OUTPUT_PREFIX="test_evaluation_"

# Load a results CSV then begin parsing its models
readarray -t array1 < <(cut -d, -f2 $CKPTCSV | awk '{if(NR>1)print}')
readarray -t array2 < <(cut -d, -f2 $MODELCSV | awk '{if(NR>1)print}')
echo $array1
echo $array2
for experiment in "${EXPERIMENTS[@]}"
do
    OUTPUT=$OUTPUT_PREFIX$experiment
    echo "Saving data in $OUTPUT"
    # for i in "${eCollection[@]}"
    for (( i=0; i<${#array1[@]}; i++ ));
    do
        echo "Beginning $i, command: CUDA_VISIBLE_DEVICES=$GPU python run_job.py --experiment=$experiment --ckpt=${array1[$i]} --model=${array2[$i]} --out_dir=$OUTPUT --placeholders --test --no_db"
        CUDA_VISIBLE_DEVICES=$GPU python run_job.py --experiment=$experiment --ckpt=${array1[$i]} --model=${array2[$i]} --out_dir=$OUTPUT --placeholders --test --no_db
    done
done

