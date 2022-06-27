#!/usr/bin/env bash

jupyter notebook \
        --NotebookApp.allow_origin='https://colab.research.google.com' \
        --port=8888 \
        --NotebookApp.port_retries=0