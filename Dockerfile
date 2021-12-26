FROM jupyter/base-notebook
# FROM continuumio/miniconda3

# expose klive and jupyter notebook ports
EXPOSE 8082
EXPOSE 8888

COPY . /home/jovyan/gdfactory
COPY docs/notebooks /home/jovyan/notebooks

RUN conda init bash

# RUN apt update
# RUN apt install gcc

RUN conda install -c conda-forge gdspy -y
RUN conda install -c conda-forge triangle -y
RUN conda install -c conda-forge pymeep -y
RUN pip install gdsfactory[full]

# COPY requirements.txt /opt/app/requirements.txt
# COPY requirements_dev.txt /opt/app/requirements_dev.txt
# WORKDIR /opt/app
# RUN pip install -r requirements.txt
# RUN pip install -r requirements_dev.txt

WORKDIR /home/jovyan
