#!/bin/bash

echo "Running BLUE" 1>&2
cd 2_BLEU
python3 demo.py > ../demo.log

echo "Running EMD based metrics" 1>&2
cd ../3_SMS
python3 demo.py >> ../demo.log

echo "Running BERTScore" 1>&2
cd ../1_BERTScore
python3 main_demo.py >> ../demo.log

echo "Running ROUGE" 1>&2
cd ../4_ROUGE
python3 main_demo.py >> ../demo.log

echo "Done" 1>&2