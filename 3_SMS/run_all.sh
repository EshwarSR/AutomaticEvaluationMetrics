#!/bin/bash

python3 -u driver_cnn.py roberta-large sms > logs/redirect_roberta_cnn_sms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1

python3 -u driver.py roberta-large wms > logs/redirect_roberta_aes_wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
python3 -u driver.py roberta-large sms > logs/redirect_roberta_aes_sms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
python3 -u driver.py roberta-large s+wms > logs/redirect_roberta_aes_s+wms_`date +%d_%m_%Y_%H_%M_%S`.log 2>&1
