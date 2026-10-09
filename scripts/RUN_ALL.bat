@echo off
rem nara_bid_stat: audit + backtest + consulting report in one run.
rem Usage:  scripts\RUN_ALL.bat [CBF.xlsx path] [prebid folder] [output folder]
rem Defaults: data\CBF.xlsx or NARA_CBF; prebid folder: NARA_PREBID, else a prebid folder next to the CBF file; results
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "ROOT=%~dp0.."
if exist "%ROOT%\nara_bid_stat\__main__.py" goto have_pkg
echo [ERROR] The nara_bid_stat folder was not found next to the scripts folder: %ROOT%
set "RC=1"
goto end
:have_pkg
set "PY=py -3.11"
%PY% -c "import sys" >nul 2>&1 || set "PY=python"
%PY% -c "import sys" >nul 2>&1 && goto have_py
echo [ERROR] Python was not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
set "RC=1"
goto end
:have_py
pushd "%ROOT%"
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 || %PY% -m pip install -r requirements.txt
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 && goto have_deps
echo [ERROR] Installing numpy, pandas and openpyxl failed. Check the internet connection or proxy and run again.
set "RC=1"
goto done
:have_deps
set "ARGS="
if not "%~1"=="" set ARGS=--cbf "%~1"
if "%~1"=="" if not defined NARA_CBF set ARGS=--cbf "%ROOT%\data\CBF.xlsx"
if not "%~2"=="" set ARGS=%ARGS% --prebid-dir "%~2"
set "OUT=%~3"
if "%OUT%"=="" set "OUT=%ROOT%\results"
%PY% -m nara_bid_stat run-all %ARGS% --out "%OUT%"
set "RC=%ERRORLEVEL%"
echo.
echo Finished with exit code %RC%. Results: %OUT%
:done
popd
:end
pause
endlocal & exit /b %RC%
