#!/bin/sh
#SBATCH --job-name=bert_wms # Job name
#SBATCH --ntasks=64 
#SBATCH --output=logs/batch_bert_wms_%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G

#LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH

#source /home/eshwarsr/IISc-ML-Project/virtual_env/bin/activate

conda activate base
python -u driver_wmt.py ../data/WMT18Data/system-outputs/newstest2018/tr-en/ ../data/WMT18Data/references/newstest2018-tren-ref.en bert wms > logs/redirect_bert_wmt_tr_wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1

echo "Done"
