@echo off
rem Windows CMD entry: runs run.ps1 (PowerShell 7 if installed, else Windows PowerShell).
rem Usage: run.cmd   or   run.cmd -Mode prod
setlocal
where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
)
exit /b %ERRORLEVEL%
