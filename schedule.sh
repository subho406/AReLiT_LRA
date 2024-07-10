#!/bin/bash

for seed in {0..14}
do
  sbatch arelit_imdb.sh $seed
  # sbatch arelit_listops.sh $seed
done