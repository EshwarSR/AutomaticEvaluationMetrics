#!/bin/bash


# python3 -u driver.py roberta-large wms > logs/redirect_roberta_sas_wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
# echo 'Done with Roberta sas wms'
# python3 -u driver.py roberta-large sms > logs/redirect_roberta_sas_sms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
# echo 'Done with Roberta sas sms'
# python3 -u driver.py roberta-large s+wms > logs/redirect_roberta_sas_s+wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
# echo 'Done with Roberta sas s+wms'

python3 -u driver_sms.py > logs/redirect_sentbert_sas_sms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
echo 'Done with SentBERT sas sms'
