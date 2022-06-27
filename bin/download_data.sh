#!/usr/bin/env bash
# Script to download, unpack, and move data into place

# Download MSR-VTT

# Videos
cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip MSRVTT.zip
cd ..

# Captions/splits/etc
cd data
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
unzip msrvtt_data.zip
cd ..

# Download MSVD

# Videos
mkdir data/msvd_videos
cd data/msvd_videos
wget https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
tar xvf YouTubeClips.tar
cd ../..

# Captions/splits/etc
cd data
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip
unzip msvd_data.zip
cd ..

# This data is stored/downloaded from git lfs
# cd data
# wget fire_aggregated_dataset.json
# wget fire_msrvtt_dataset.json
# wget fire_msvd_dataset.json
# cd ..

# wget predictions.zip
# unzip predictions.zip
