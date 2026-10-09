@echo off
rem nara_bid_stat: audit + backtest + consulting report in one run.
rem Usage:  scripts\RUN_ALL.bat [CBF.xlsx path] [prebid folder] [output folder]
rem Defaults: data\CBF.xlsx, data\prebid, results
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "ROOT=%~dp0.."
if exist "%ROOT%\nara_bid_stat\__main__.py" goto have_pkg
echo [ERROR] The nara_bid_stat folder was not found next to the scripts folder: %ROOT%
set "RC=1"
goto end
:have_pkg
set "CBF=%~1"
if "%CBF%"=="" set "CBF=%ROOT%\data\CBF.xlsx"
set "PREBID=%~2"
if "%PREBID%"=="" set "PREBID=%ROOT%\data\prebid"
set "OUT=%~3"
if "%OUT%"=="" set "OUT=%ROOT%\results"
set "PY=py -3.11"
%PY% -c "import sys" >nul 2>&1 || set "PY=python"
%PY% -c "import sys" >nul 2>&1 && goto have_py
echo [ERROR] Python was not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
set "RC=1"
goto end
:have_py
pushd "%ROOT%"
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 || %PY% -m pip install -r requirements.txt
set "PBARG="
if exist "%PREBID%\" set PBARG=--prebid-dir "%PREBID%"
if not exist "%PREBID%\" echo [WARN] prebid folder not found: %PREBID% - running without prebid data.
%PY% -m nara_bid_stat run-all --cbf "%CBF%" %PBARG% --out "%OUT%"
set "RC=%ERRORLEVEL%"
popd
echo.
echo Finished with exit code %RC%. Results: %OUT%
:end
pause
endlocal & exit /b %RC%
