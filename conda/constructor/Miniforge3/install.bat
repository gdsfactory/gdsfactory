
cd ..\condabin
call conda activate

call pip install gdsfactory[full]==6.115.0
call conda install -c conda-forge slepc4py=*=complex* -y
call conda install -c conda-forge git -y
call pip install femwell
