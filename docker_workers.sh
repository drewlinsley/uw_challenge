#!/bin/bash

echo "Which node do you want to use? [1, 2, 3, 4, 5, 6]"
read node

for gpu in 0 1 2 3 4 5 6 7
# for gpu in 0 1 2 3 4 5
# for gpu in 1 3
# for gpu in 4 5 6 7
# for gpu in 1 2 7
# for gpu in 7
# for gpu in 6 7
# for gpu in 0
do
    echo "GPU $gpu"
    # Build the image and label it on the Docker registry
    if [ $gpu = 0 ]
    then
        nvidia-docker build -t serrep$node.services.brown.edu:5000/cluttered_nist_experiments .
    fi

    #Run the container
    nvidia-docker run -d --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs/cluster_projects/cluttered_nist_experiments serrep$node.services.brown.edu:5000/cluttered_nist_experiments bash start_worker.sh $gpu
done
