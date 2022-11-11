
@REM set PIP_FIND_LINKS="https://whls.blob.core.windows.net/unstable/index.html"
@REM pip install lytest simphony sax jax sklearn klayout devsim
@REM pip install "jaxlib[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
@REM pip install gdsfactory==5.54.0
@REM gf tool install
@REM cd ..
set GF_PATH=%cd%
echo %GF_PATH%

@REM if exist "%USERPROFILE%\Desktop\gdsfactory" (goto SKIP_INSTALL)
cd %USERPROFILE%\Desktop
@REM pause
call %GF_PATH%\condabin\conda activate
@REM pause
call conda install -c conda-forge git -y
@REM pause
call git clone https://github.com/gdsfactory/gdsfactory.git
@REM pause
echo %GF_PATH%

cd gdsfactory
%GF_PATH%\python shortcuts.py %GF_PATH%

:SKIP_INSTALL
echo gdsfactory installed
