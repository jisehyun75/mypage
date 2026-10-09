@echo off
rem nara_bid_stat: analyze notices -> BEST10 + recommended bid rate, amount and win probability.
rem Usage:  scripts\ANALYZE.bat [notice numbers ...]
rem   - no argument: asks for notice numbers (separate several with spaces)
rem   - type LIST at the prompt to see the notices in CBF that are not opened yet
rem Data:   data\CBF.xlsx and data\prebid (or set NARA_CBF / NARA_PREBID to other paths)
rem Output: results\analyze\
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "ROOT=%~dp0.."
set "CBF=%NARA_CBF%"
if "%CBF%"=="" set "CBF=%ROOT%\data\CBF.xlsx"
set "PREBID=%NARA_PREBID%"
if "%PREBID%"=="" set "PREBID=%ROOT%\data\prebid"
set "PY=py -3.11"
%PY% -c "import sys" >nul 2>&1 || set "PY=python"
pushd "%ROOT%"
%PY% -c "import numpy, pandas, openpyxl" >nul 2>&1 || %PY% -m pip install -r requirements.txt
set "PBARG="
if exist "%PREBID%\" set PBARG=--prebid-dir "%PREBID%"
set "NOTICES=%*"
:ask
if not "%NOTICES%"=="" goto run
set /p "NOTICES=Notice number(s) (LIST = show pending notices, Enter = quit): "
if "%NOTICES%"=="" goto done
:run
if /i "%NOTICES%"=="LIST" (
  %PY% -m nara_bid_stat analyze --cbf "%CBF%" --list
  set "NOTICES="
  goto ask
)
%PY% -m nara_bid_stat analyze --cbf "%CBF%" %PBARG% --notice %NOTICES% --out "%ROOT%\results\analyze"
set "NOTICES="
if "%~1"=="" goto ask
:done
popd
pause
endlocal
