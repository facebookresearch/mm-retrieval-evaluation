# Multimodal Retrieval Evaluation

## Reproducing Results

Reproducing our experiments requires several steps:

1. Install the dependent software, see instructions below
2. Download the pre-requisite data to run our experiments
3. Running our experiment pipeline

The following sections discuss each step

### Software

To install the software, follow these steps:

1. Install [python poetry](https://python-poetry.org/) using system package manager or default installation command `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
2. Install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setup the environment so that `conda` runs
3. Create a python environment with `conda create -n mmr python=3.9`
4. Activate the environment with `conda activate mmr`
5. Install the required packages with `poetry install`
6. (Optional, only needed if generating PDF plots): Install Altair deps: `conda install -c conda-forge vega-cli vega-lite-cli`

These installation instructions should set things up for either the code repository (this one) or the paper repository (on overleaf).

### Download Data

Data is obtained via two sources, [Git LFS](https://git-lfs.github.com/) to provide access to data directly associated with this paper and a download script for external datasets.
Make sure to have Git LFS properly installed and that when you clone, there are data files in `data` like `fire_aggregated_dataset.json`
All data is stored in `data` and experimental code writes to various paths in this directory.
Following this, run the download script via `bash bin/download_data.sh`.
NOTE: This will download the MSR-VTT and MSVD videos, which are several GB in size. If you have these on your system already, you can reference the script to identify where to copy/simlink to.

### Running the Experiment Pipeline

We manage our experiments using [Luigi](https://github.com/spotify/luigi), which is a make-like python framework.
In essence, tasks correspond to particular experimental steps (e.g., run a model or format a dataset) and these tasks are assembled into a tree/DAG.
Running the tasks leaf tasks in `mmr/tasks.py` reproduces the paper experiments.
To fully reproduce the paper, you can run the sequence of commands after changing `--workers 0` to `--workers ` (or any number greater than 0).
Running with zero workers is a good way to test if the data downloads were put in the correct spot.

```
luigi --module mmr.tasks --local-scheduler --workers 0 AllSimToPreds
luigi --module mmr.tasks --local-scheduler --workers 0 AllAutomaticMsrvttEvals 
luigi --module mmr.tasks --local-scheduler --workers 0 AllAutomaticMsrvttMatchingLabels
luigi --module mmr.tasks --local-scheduler --workers 0 AllModelEvaluations
```

After letting this run, the `data` directory should be a reproduction of our experimental runs.

We will add additional instructions here when the repository containing the source latex and code for the paper PDF is open sourced.

## License

The code and data in this work are licensed under the [Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0) license](https://creativecommons.org/licenses/by-nc/2.0/).