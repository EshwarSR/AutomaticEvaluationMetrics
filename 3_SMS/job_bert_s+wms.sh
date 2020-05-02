#!/bin/sh
#SBATCH --job-name=bert_s+wms # Job name
#SBATCH --ntasks=64 
#SBATCH --output=logs/batch_bert_s+wms_%j.out # Standard output and error log
#SBATCH --partition=cl2_48h-1G

#LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH

#source /home/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

conda activate base
python -u driver_wmt.py ../data/WMT18Data/system-outputs/newstest2018/et-en/ ../data/WMT18Data/references/newstest2018-eten-ref.en bert s+wms > logs/redirect_bert_wmt_et_s+wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1

echo "Done"
