#!/usr/bin/env bash

zip predictions.zip \
    data/predictions/msrvtt/clip4clip/sim_test.npy \
    data/predictions/msrvtt/collaborative-experts/sim_test.npy \
    data/predictions/msrvtt/ssb/sim_test.npy \
    data/predictions/msvd/clip4clip/sim_test.npy \
    data/predictions/msvd/collaborative-experts/sim_test.npy \
    data/predictions/msvd/collaborative-experts/meta.pkl
