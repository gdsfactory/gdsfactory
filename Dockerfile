FROM continuumio/miniconda3
# FROM jupyter/base-notebook

COPY environment.yml ./
RUN conda env create -f environment.yml

RUN conda install -c conda-forge gdspy
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt
