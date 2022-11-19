
set PIP_FIND_LINKS="https://whls.blob.core.windows.net/unstable/index.html"
pip install lytest simphony sax jax sklearn klayout devsim
pip install "jaxlib[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install gdsfactory==6.1.1
gf tool install

cd ..\condabin
call conda activate
call conda install -c conda-forge git -y
if exist "%USERPROFILE%\Desktop\gdsfactory" (goto SKIP_INSTALL)
cd %USERPROFILE%\Desktop
call git clone https://github.com/gdsfactory/gdsfactory.git

:SKIP_INSTALL
echo gdsfactory installed

cd gdsfactory
python shortcuts.py
