\
@echo off
cd /d %~dp0
python src\train_tokenizer.py --text data\train.txt --vocab_size 1000
pause
