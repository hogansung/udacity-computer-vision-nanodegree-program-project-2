FROM nvidia/cuda:11.2.2-base-ubuntu20.04
CMD nvidia-smi

RUN apt-get update && apt-get install -y curl python3 python3-pip vim
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN alias python=python3.8

COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN mkdir /app/models
VOLUME /opt/cocoapi

WORKDIR /app/src
