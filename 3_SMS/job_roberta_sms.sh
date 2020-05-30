#!/bin/sh
#SBATCH --job-name=rb_sms # Job name
#SBATCH --ntasks=64 
#SBATCH --output=logs/batch_roberta_sms_%j.out # Standard output and error log
#SBATCH --partition=cl1_all_64C

#LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH

#source /home/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

conda activate base
python -u driver.py roberta-large sms > logs/redirect_roberta_aes_sms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
echo "Done"
