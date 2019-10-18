time=$(date "+%Y%m%d-%H%M%S")
#logname="../logs/${time}.log"
nohup /home/jbj/anaconda3/bin/python3 train.py > log.out 2>&1 &
import torch
t