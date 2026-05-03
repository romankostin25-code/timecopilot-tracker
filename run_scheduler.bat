@echo off
REM Activate virtual environment and run the scheduler (Windows)
SET DIR=%~dp0
CALL "%DIR%.venv\Scripts\activate.bat"
python "%DIR%scheduler.py"
