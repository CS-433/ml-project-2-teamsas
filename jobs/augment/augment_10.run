#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 3:00:00

python main_data.py -i data/dataset.xlsx -o artifacts/data --clean --n_aug 1 --translation --target_lang french --noise_injection --char_insert_aug_p 0.2 --ocr_aug_p 0.1 --word_swapping_aug_p 0.2 --tf_idf_dropping_p 0.3 --tf_idf_syn_replace_p 0.5 --target_corpus reuters --masking_p 0.5 --model_name bert-base-uncased --row_start 1800 --row_end 1998