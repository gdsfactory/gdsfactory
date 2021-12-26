FROM jupyter/base-notebook
# FROM continuumio/miniconda3

# expose klive and jupyter notebook ports
EXPOSE 8082
EXPOSE 8888
COPY . .

RUN conda init bash

# RUN apt update
# RUN apt install gcc
# Activate the environment, and make sure it's activated:
# RUN echo "conda activate myenv" > ~/.bashrc

RUN conda install -c conda-forge gdspy -y
RUN conda install -c conda-forge triangle -y
RUN conda install -c conda-forge pymeep -y
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt
