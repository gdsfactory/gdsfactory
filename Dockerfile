FROM continuumio/miniconda3
# FROM jupyter/base-notebook

EXPOSE 8082
COPY . .

RUN conda init bash
RUN apt install gcc

# Activate the environment, and make sure it's activated:
# RUN echo "conda activate myenv" > ~/.bashrc

RUN conda install -y gdspy
RUN conda install -y triangle
RUN bash install.sh
