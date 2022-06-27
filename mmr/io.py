from typing import Union, List, Any
import platform
import subprocess
import os
from pathlib import Path
from pydantic import BaseModel
import simdjson
import altair as alt
import altair_saver


def read_json(path: Union[str, Path]):
    """
    Read a json file from a string path
    """
    with open(path) as f:
        return simdjson.load(f)


def safe_file(path: Union[str, Path]) -> Union[str, Path]:
    """
    Ensure that the path to the file exists, then return the path.
    For example, if the path passed in is /home/entilzha/stuff/stuff/test.txt,
    this function will run the equivalent of mkdir -p /home/entilzha/stuff/stuff/
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_chart(
    chart: alt.Chart, base_path: Union[Path, str], filetypes: List[str], method=None
):
    base_path = str(base_path)
    for t in filetypes:
        path = base_path + "." + t
        if method == "node" and t in ("svg", "pdf"):
            method = "node"
        else:
            method = None
        altair_saver.save(chart, safe_file(path), method=method)


def md5sum(filename: str) -> str:
    system = platform.system()
    if system == "Linux":
        command = f"md5sum {filename}"
    elif system == "Darwin":
        command = f"md5 -r {filename}"
    else:
        raise ValueError(f"Unexpected platform for md5: {system}")
    return (
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, check=True)
        .stdout.decode("utf-8")
        .split()[0]
    )


def write_json(path: Union[str, Path], obj: Any):
    """
    Write an object to a string path as json.
    If the object is a pydantic model, export it to json
    """
    if isinstance(obj, BaseModel):
        with open(path, "w") as f:
            f.write(obj.json())
    else:
        with open(path, "w") as f:
            simdjson.dump(obj, f)
