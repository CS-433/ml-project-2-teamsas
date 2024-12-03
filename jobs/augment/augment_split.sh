#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 3:00:00
#SBATCH --mem 4096
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

echo "job $1 started"
mkdir -p $HOME/ml-project-2-teamsas/artifacts/data/subset_$1

echo start time: $(date)
echo start index: $((($1 - 1) * 200))
echo end index: $(($1 * 200))

python $HOME/ml-project-2-teamsas/main_data.py \
    -i $HOME/ml-project-2-teamsas/data/dataset.xlsx \
    -o $HOME/ml-project-2-teamsas/artifacts/data/subset_$1 \
    --clean \
    --n_aug 1 \
    --translation \
    --target_lang french \
    --noise_injection \
    --char_insert_aug_p 0.2 \
    --ocr_aug_p 0.1 \
    --word_swapping_aug_p 0.2 \
    --tf_idf_dropping_p 0.3 \
    --tf_idf_syn_replace_p 0.5 \
    --target_corpus reuters \
    --masking_p 0.5 \
    --model_name bert-base-uncased \
    --row_start $((($1 - 1) * 200)) \
    --row_end $(($1 * 200)) \

echo "job $1 finished"
