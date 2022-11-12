
set PIP_FIND_LINKS="https://whls.blob.core.windows.net/unstable/index.html"
pip install lytest simphony sax jax sklearn klayout devsim
pip install "jaxlib[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install gdsfactory==6.0.0
gf tool install
cd ..
set GF_PATH=%cd%

if exist "%USERPROFILE%\Desktop\gdsfactory" (goto SKIP_INSTALL)
cd %USERPROFILE%\Desktop
git clone https://github.com/gdsfactory/gdsfactory.git

cd gdsfactory
%GF_PATH%\python shortcuts.py %GF_PATH%

:SKIP_INSTALL
echo gdsfactory installed
