
pip install lytest simphony sax sklearn klayout
pip install gdsfactory==5.33.10
gf tool install

if exist "%USERPROFILE%\Desktop\gdsfactory" (goto SKIP_INSTALL)
cd %USERPROFILE%\Desktop
git clone https://github.com/gdsfactory/gdsfactory.git

:SKIP_INSTALL
echo gdsfactory installed
