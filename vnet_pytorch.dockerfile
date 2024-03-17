FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set bash as the default shell
ENV SHELL=/bin/bash
ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"
ENV HTTP_PROXY="http://proxy.san.gva.es:8080"
ENV HTTPS_PROXY="http://proxy.san.gva.es:8080"
ENV FTP_PROXY="http://proxy.san.gva.es:8080"

# Create a working directory
WORKDIR /vnet/

RUN mkdir /data/
RUN mkdir /vnet/results/

COPY train_vnet.py /vnet/
COPY engine.py /vnet/
COPY vnet_model.py /vnet/
COPY data_setup.py /vnet/
COPY metrics.py /vnet/
COPY utils.py /vnet/
COPY config/vnet_config.yaml /vnet/config/vnet_config.yaml

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# build with some basic python packages
RUN pip install \
    numpy \
    pandas==2.1.3 \
    tqdm \
    Pillow \
    scikit-image==0.22.0 \
    torchinfo==1.8.0 \
    torchmetrics==1.2.0 \
    matplotlib==3.8.2 \
    SimpleITK==2.3.1

CMD ["bash"]
