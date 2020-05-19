#!/bin/sh
#SBATCH --job-name=elmo_wms # Job name
#SBATCH --ntasks=64 
#SBATCH --output=logs/batch_elmo_wms_%j.out # Standard output and error log
#SBATCH --partition=cl2_48h-1G

#LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH

#source /home/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

conda activate base
python -u driver.py elmo wms > logs/redirect_elmo_aes_wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
echo "Done"
