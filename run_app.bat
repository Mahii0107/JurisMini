@echo off
title JurisMini - Neobrutalist Legal SLM Explorer
echo ===================================================
echo   JurisMini: Judicial SLM Intelligence (Neobrutalist)
echo ===================================================
echo.
echo [1/2] Starting Python Legal AI Backend on port 5000...
start /b python server.py
echo [2/2] Starting React + Vite Frontend on port 5173...
cd /d "%~dp0frontend"
call npm run dev
pause
