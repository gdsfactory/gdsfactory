
set PIP_FIND_LINKS="https://whls.blob.core.windows.net/unstable/index.html"

cd ..\condabin
call conda activate

call pip install sax jax sklearn
call pip install "jaxlib[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
call pip install gdsfactory[full,dev,gmsh,tidy3d,devsim,meow,ray,database,kfactory]==6.104.2

call conda install -c conda-forge slepc4py=*=complex* -y
call conda install -c conda-forge git -y
call pip install femwell
