#!/bin/sh
#SBATCH --job-name=aes # Job name
#SBATCH --ntasks=64 # Run on a eight CPU

#SBATCH --output=logs/batch_aes_%j.out # Standard output and error log
#SBATCH --partition=cl1_all_64C

source /scratch/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

python -u dump_spacy_objs.py  asap_aes.xlsx  >logs/redirect_aes_`date +%H_%M_%S_%d_%m_%Y`.log 2>&1

echo "Done"