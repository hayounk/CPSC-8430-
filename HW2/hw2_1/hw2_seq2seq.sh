#!/bin/bash
wget https://drive.google.com/u/0/uc?id=1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ&export=download

tar zxvf MLDS_hw2_1_data.tar.gz

python seq2seq.py $1 $2 --test
