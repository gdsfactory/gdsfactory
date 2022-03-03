rem ensures conda is installed and installs gdsfactory package
rem https://gist.github.com/nimaid/a7d6d793f2eba4020135208a57f5c532
rem https://gist.github.com/maximlt/531419545b039fa33f8845e5bc92edd6
@echo off

set ORIGDIR="%CD%"

set MINICONDAPATH=%USERPROFILE%\Miniconda3
set CONDAEXE=%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%-condainstall.exe
set "OS="
set "MCLINK="

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 goto CONDAFOUND

:INSTALLCONDA
reg Query "HKLM\Hardware\Description\System\CentralProcessor\0" | find /i "x86" > NUL && set OS=32BIT || set OS=64BIT
if %OS%==32BIT set MCLINK=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe
if %OS%==64BIT set MCLINK=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

echo Downloading Miniconda3 (This will take while, please wait)...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%MCLINK%', '%CONDAEXE%')" >nul 2>nul
if errorlevel 1 goto CONDAERROR

echo Installing Miniconda3 (This will also take a while, please wait)...
start /wait /min "Installing Miniconda3..." "%CONDAEXE%" /InstallationType=JustMe /S /D="%MINICONDAPATH%"
del "%CONDAEXE%"
if not exist "%MINICONDAPATH%\" (goto CONDAERROR)

"%MINICONDAPATH%\Scripts\conda.exe" init
if errorlevel 1 goto CONDAERROR

echo Miniconda3 has been installed!

:CONDAFOUND
echo Conda is already installed!

call conda create -n gdsfactory -y
call conda activate gdsfactory
call conda install -c conda-forge gdspy -y
call pip install -e .[dev] --upgrade

git clone https://github.com/gdsfactory/gdslib.git -b data
pre-commit install
gf tool install


:END
exit /B 0

:CONDAERROR
echo Miniconda3 install failed!
exit /B 1
