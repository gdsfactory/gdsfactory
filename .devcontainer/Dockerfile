FROM python:3.12-slim

# Expose the necessary ports
EXPOSE 8082 8083 8888

# Run as root to install packages
USER root
COPY  --chmod=777 . ${HOME}

# Update and install dependencies
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        curl \
        git \
        htop \
        klayout \
        make \
        neovim \
        python3-gmsh \
        gnupg && \
    rm -rf /var/lib/apt/lists/*

USER ${USER}
# Using pip to install Python packages
# Ensure your base image has all the necessary libraries or install them before switching users
RUN pip install .[dev,docs,full]
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
