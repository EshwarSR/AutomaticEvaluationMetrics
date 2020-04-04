#!/bin/sh
#SBATCH --job-name=sas # Job name
#SBATCH --ntasks=64 # Run on a eight CPU

#SBATCH --output=logs/batch_sas_%j.out # Standard output and error log
#SBATCH --partition=cl2_48h-1G

source /scratch/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

python -u dump_spacy_objs.py  asap_sas.tsv  >logs/redirect_sas_`date +%H_%M_%S_%d_%m_%Y`.log 2>&1

echo "Done"