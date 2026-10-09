@echo off
rem nara_bid_stat: analyze notices -> BEST10 + recommended bid rate, amount and win probability.
rem Usage:  scripts\ANALYZE.bat [notice numbers ...]
rem   - no argument: asks for notice numbers, separate several with spaces
rem   - type LIST at the prompt to see the notices in CBF that are not opened yet
rem Data:   data\CBF.xlsx and data\prebid, or set NARA_CBF / NARA_PREBID to other paths
rem Output: results\analyze\
rem Keep this file inside the scripts folder. For a desktop icon: right-click, Send to, Desktop - create shortcut.
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "ROOT=%~dp0.."
if exist "%ROOT%\nara_bid_stat\__main__.py" goto have_pkg
echo [ERROR] The nara_bid_stat folder was not found next to the scripts folder: %ROOT%
echo         Keep ANALYZE.bat inside the scripts folder. For the desktop, create a shortcut instead of moving it.
goto end
:have_pkg
set "PY=py -3.11"
%PY% -c "import sys" >nul 2>&1 || set "PY=python"
%PY% -c "import sys" >nul 2>&1 && goto have_py
echo [ERROR] Python was not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
goto end
:have_py
pushd "%ROOT%"
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 || %PY% -m pip install -r requirements.txt
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 && goto have_deps
echo [ERROR] Installing numpy, pandas and openpyxl failed. Check the internet connection or proxy and run again.
goto done
:have_deps
rem The prebid folder is found by Python: NARA_PREBID, else a folder named prebid (or the Korean prebid name) next to the CBF file.
set "ARGS="
if not defined NARA_CBF set ARGS=--cbf "%ROOT%\data\CBF.xlsx"
set "NOTICES=%*"
:ask
if not "%NOTICES%"=="" goto run
set /p "NOTICES=Notice number(s) - LIST shows pending notices, Enter quits: "
if "%NOTICES%"=="" goto done
:run
if /i not "%NOTICES%"=="LIST" goto analyze
%PY% -m nara_bid_stat analyze %ARGS% --list
set "NOTICES="
goto ask
:analyze
%PY% -m nara_bid_stat analyze %ARGS% --notice %NOTICES% --out "%ROOT%\results\analyze"
set "NOTICES="
if "%~1"=="" goto ask
:done
popd
:end
pause
endlocal
