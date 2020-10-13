rem this batch file checks for python version is 3x.
rem and starts the flask api

rem check python version 3x or not
for /f "delims=" %%i in ('python --version') do (
set pyversion=%%i
)
%%pyversion 3>NUL
if not errorlevel 0 ...
rem check current path
set apipath=%cd%
echo %apipath%
echo "install requirements"
pip install -r requirements.txt
python run.py