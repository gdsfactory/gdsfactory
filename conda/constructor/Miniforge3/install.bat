
cd ..\condabin
call conda activate

call pip install gdsfactory[cad]==7.4.3 gplugins
call conda install -c conda-forge slepc4py=*=complex* -y
call conda install -c conda-forge git -y
call pip install femwell
