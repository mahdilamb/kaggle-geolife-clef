
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt -yq update && apt install -yq software-properties-common git && apt -yq update && apt -yqq install ssh curl
ENV DEBIAN_FRONTEND=noninteractive 

RUN add-apt-repository ppa:deadsnakes/ppa && apt update && apt -yqq install python3.11 python3.11-distutils && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && update-alternatives --install /usr/bin/python3 python3 $(which python3.11) 1 &&update-alternatives --config python3 && python3 -m pip install requests packaging pip-tools


WORKDIR /app
COPY geolife_clef_2024/__init__.py geolife_clef_2024/__init__.py
COPY requirements.in .
COPY pyproject.toml .
COPY Makefile .

RUN git config --global url."https://github.com/mahdilamb".insteadOf "ssh://git@github.com/mahdilamb" && make requirements && make requirements extras=all
RUN python3.11 -m pip install -r requirements-all.txt --no-cache $(make pip-args)

COPY geolife_clef_2024 geolife_clef_2024

RUN python3.11 -m pip install . --no-dependencies

