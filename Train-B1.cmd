@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0experiments\b1\deployment\Start-B1.ps1" %*
pause
