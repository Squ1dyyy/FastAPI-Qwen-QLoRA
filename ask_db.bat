@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
python ask_db.py %*
