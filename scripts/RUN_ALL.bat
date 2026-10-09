@echo off
rem nara_bid_stat: audit + backtest + consulting report in one run.
rem Usage:  scripts\RUN_ALL.bat [CBF.xlsx path] [prebid folder] [output folder]
rem Defaults: data\CBF.xlsx, data\prebid, results
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "ROOT=%~dp0.."
set "CBF=%~1"
if "%CBF%"=="" set "CBF=%ROOT%\data\CBF.xlsx"
set "PREBID=%~2"
if "%PREBID%"=="" set "PREBID=%ROOT%\data\prebid"
set "OUT=%~3"
if "%OUT%"=="" set "OUT=%ROOT%\results"
set "PY=py -3.11"
%PY% -c "import sys" >nul 2>&1 || set "PY=python"
pushd "%ROOT%"
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 || %PY% -m pip install -r requirements.txt
if exist "%PREBID%" (
  %PY% -m nara_bid_stat run-all --cbf "%CBF%" --prebid-dir "%PREBID%" --out "%OUT%"
) else (
  %PY% -m nara_bid_stat run-all --cbf "%CBF%" --out "%OUT%"
)
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
