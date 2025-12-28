\
@echo off
cd /d %~dp0
python src\prepare_medical_corpus.py --input_txt data\raw\medical.txt --out_txt data\train.txt
pause
