#!/bin/sh
DOWNLOAD_DIR=./data
COMPETITION=geolifeclef-2024

if [ ! -f 'data/GLC24_P0_metadata_train.csv' ]; then
    python3 -c 'import kaggle' || python3 -m pip install kaggle
    kaggle competitions download -c $COMPETITION
    unzip -o $COMPETITION.zip -d $DOWNLOAD_DIR
    rm -rf $COMPETITION.zip
fi

python3 -c 'import seafile_downloader' || python3 -m pip install 'seafile-downloader @ git+ssh://git@github.com/mahdilamb/seafile-downloader@v0.1.0'
python3 -m seafile_downloader 'https://lab.plantnet.org/seafile/d/bdb829337aa44a9489f6/' --out ./data/extra --timeout 3600 --retry 5
