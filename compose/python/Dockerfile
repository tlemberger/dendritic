FROM nvcr.io/nvidia/pytorch:23.12-py3 AS build-image
# FROM nvcr.io/nvidia/pytorch:22.01-py3 AS build-image
# FROM python:3.8-buster as build-image
# FROM arm64v8/python:3.8 as build-image
# FROM python:3.10 AS compile-image-amd64

# ARG TARGETARCH
# FROM compile-image AS build-image

RUN apt-get update -y &&  DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv
RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

# Install any dependencies from setup.py
# RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake

COPY setup.py setup.py
# run pip install to upgrade and then to install
RUN pip install --upgrade pip && pip install -e .
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# download nltk punkt tokenization file
RUN python -c "import nltk; nltk.download('punkt')"
# Download spacy model
RUN python -m spacy download en_core_web_sm
# copy JATS dtd
COPY JATS-journalpublishing1.dtd /app/JATS-journalpublishing1.dtd

# Set the working directory in the container
WORKDIR /app

EXPOSE 8889/tcp
