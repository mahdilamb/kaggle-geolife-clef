#!/bin/sh
DOWNLOAD_DIR=./data
COMPETITION=geolifeclef-2024

if [ ! -f 'data/GLC24_P0_metadata_train.csv' ]; then
    python3 -c 'import kaggle' || python3 -m pip install kaggle
    kaggle competitions download -c $COMPETITION
    unzip -o $COMPETITION.zip -d $DOWNLOAD_DIR
    rm -rf $COMPETITION.zip
fi
