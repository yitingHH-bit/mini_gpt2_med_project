\
@echo off
cd /d %~dp0
python src\generate.py --prompt "Explain what screening means for cancer." --max_new_tokens 120
pause
