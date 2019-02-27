#!/bin/bash
# GPU worker for running experiments in your database. Continues until the DB is empty.

if [ -z $1 ]
then
    read -p "Enter the ID of the gpu you want to use: "  gpu
else
    gpu=$1
fi
echo "Developing worker for gpu $gpu."

export PGPASSWORD=serrelab
RUN_LOOP=true
while [ $RUN_LOOP == "true" ]
    do
        CHECK="$(psql -U contextual_DCN -h 127.0.0.1 -d contextual_DCN -c 'SELECT * from experiments h WHERE NOT EXISTS (SELECT 1 FROM in_process i WHERE h._id = i.experiment_id)')"
        if [[ $CHECK =~ "(0 rows)" ]]; then  # EMPTY
            RUN_LOOP=false
        fi
        CUDA_VISIBLE_DEVICES=$gpu python run_job.py
    done

