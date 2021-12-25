FROM continuumio/miniconda3
# FROM jupyter/base-notebook

EXPOSE 8082
COPY . .

RUN conda install -c conda-forge gdspy -y
RUN bash install.sh
