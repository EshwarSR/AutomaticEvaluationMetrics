#!/bin/sh
#SBATCH --job-name=elmo_sms # Job name
#SBATCH --ntasks=64 
#SBATCH --output=logs/batch_elmo_sms_%j.out # Standard output and error log
#SBATCH --partition=cl1_all_64C

#LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH

#source /home/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

conda activate base
python -u driver_wmt.py ../data/WMT18Data/system-outputs/newstest2018/et-en/ ../data/WMT18Data/references/newstest2018-eten-ref.en elmo sms > logs/redirect_elmo_wmt_et_sms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1

echo "Done"
