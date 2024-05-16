
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt -yq update && apt install -yq software-properties-common git && apt -yq update && apt -yqq install ssh curl
RUN add-apt-repository ppa:deadsnakes/ppa && apt update && apt -yqq install python3.11 python3.11-distutils && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && update-alternatives --install /usr/bin/python3 python3 $(which python3.11) 1 &&update-alternatives --config python3 && python3 -m pip install requests packaging pip-tools

RUN python3 -m pip config set global.extra-index-url $(python3 -c 'import subprocess, requests, re, packaging.version;cuda_version = int("'$CUDA_VERSION'".split(".")[0]);print("" if cuda_version == -1 else "https://download.pytorch.org/whl/"+(sorted([version for version in re.findall(r"^\<a href=\"(cpu|cu\d+)\/torch-(\d+\.\d+\.\d+)",requests.get("https://download.pytorch.org/whl/torch_stable.html").text,re.MULTILINE) if version[0].startswith("cu") if int(version[0][2:][:-1]) == cuda_version ], key=lambda x: (int(x[0][2:][-1]), packaging.version.Version(x[1])),reverse=True)[0][0]))') && python3 -m pip install torch --no-cache

WORKDIR /app
COPY requirements.in requirements.txt
RUN python3 -m pip install -r requirements.txt --no-cache

COPY geolife_clef_2024/__init__.py geolife_clef_2024/__init__.py
COPY pyproject.toml .

RUN git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
RUN python3 -m pip install -e .[training] --no-cache

COPY geolife_clef_2024 geolife_clef_2024

