
pip install lytest simphony sax sklearn klayout
pip install gdsfactory==5.33.9
gf tool install

if exist "%USERPROFILE%\gdsfactory" (goto SKIP_INSTALL)
cd %USERPROFILE%
git clone https://github.com/gdsfactory/gdsfactory.git

:SKIP_INSTALL
echo Conda is already installed!
