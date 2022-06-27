import json
import os
from pathlib import Path
from mmr.io import read_json
import streamlit as st

DATA_DIR = Path(os.environ.get("MMR_DATA", "data"))


@st.experimental_memo
def load_clusters():
    return read_json(DATA_DIR / "msrvtt-duplicate-clusters.json")


def video_id_to_url(video_id: str):
    return f"http://localhost:8000/{video_id}.mp4"


st.header("MSR VTT Duplicate Checking")

page_size = 20
clusters = load_clusters()
page = int(st.sidebar.number_input("Page", min_value=0, value=0))
st.sidebar.write(f"Showing: {page * page_size} to {(page + 1) * page_size}")
n_clusters = len(clusters["candidates"])
st.sidebar.write(f"N Clusters: {n_clusters}")


def display_clusters():
    for c in clusters["candidates"][page * page_size : (page + 1) * page_size]:
        st.write(c)
        for video_id in c:
            st.video(video_id_to_url(video_id))


display_clusters()
