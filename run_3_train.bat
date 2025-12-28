\
@echo off
cd /d %~dp0
python src\train.py --train_txt data\train.txt --epochs 10 --batch_size 16 --seq_len 64
pause
