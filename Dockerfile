FROM continuumio/miniconda3
# FROM jupyter/base-notebook

EXPOSE 8082
COPY . .

RUN conda init bash

# Activate the environment, and make sure it's activated:
# RUN echo "conda activate myenv" > ~/.bashrc

RUN conda install -c conda-forge gdspy -y
RUN bash install.sh
