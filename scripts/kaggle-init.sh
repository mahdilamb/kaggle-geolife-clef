#!/usr/bin/env bash

GIT_SHA=6ddfde7f00fc48985eec15cd53454bf179ab7280

if [[ ! $(python3 --version | grep '3.11') ]]; then
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11-full -y
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
    sudo update-alternatives --install /usr/bin/python3 python3 $(which python3.11) 1
    sudo update-alternatives --config python3
    sudo update-alternatives --install /usr/bin/python python $(which python3.11) 1
    sudo update-alternatives --config python
fi

if [[ $(basename $(pwd)) != 'kaggle-geolife-clef' ]]; then

    git config --global url."https://github.com/mahdilamb".insteadOf "ssh://git@github.com/mahdilamb"
    git clone https://github.com/mahdilamb/kaggle-geolife-clef.git
    cd kaggle-geolife-clef
    git reset --hard ${GIT_SHA}
    sudo ln -s /kaggle/input/geolifeclef-2024 ./data
    rm -f requirements*.txt
    make install-all
fi
