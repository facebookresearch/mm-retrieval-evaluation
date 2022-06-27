import os
import pickle
import subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(os.environ.get("MMR_DATA", "data"))


class MSRVTT:
    def __init__(self, js_fusion_csv: str) -> None:
        self.df = pd.read_csv(js_fusion_csv)
        self.idx_to_vid = {}
        self.vid_to_idx = {}
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            self.idx_to_vid[i] = row.video_id
            self.vid_to_idx[row.video_id] = i
        self.captions = self.df["sentence"].values
        self.video_ids = self.df["video_id"].values


class MSVD:
    def __init__(self, *, video_ids, idx_to_vid, vid_to_idx, captions) -> None:
        self.idx_to_vid = idx_to_vid
        self.vid_to_idx = vid_to_idx
        self.captions = captions
        self.video_ids = video_ids


@st.experimental_memo
def load_data():
    with open(DATA_DIR / "msvd_data/test_list.txt") as f:
        video_ids = [r.strip() for r in f]
    idx_to_vid = {idx: vid for idx, vid in enumerate(video_ids)}
    vid_to_idx = {vid: idx for idx, vid in idx_to_vid.items()}

    with open(DATA_DIR / "msvd_data/raw-captions.pkl", "rb") as f:
        msvd_captions = pickle.load(f)

    caption_map = {}
    for video_id, captions in msvd_captions.items():
        caption_map[video_id] = {}
        for idx, c in enumerate(captions):
            text = " ".join(c)
            caption_map[video_id][idx] = text

    msrvtt = MSRVTT(DATA_DIR / "msrvtt_data/MSRVTT_JSFUSION_test.csv")
    msvd = MSVD(
        idx_to_vid=idx_to_vid,
        vid_to_idx=vid_to_idx,
        captions=caption_map,
        video_ids=video_ids,
    )
    return msrvtt, msvd


MSRVTT_VIDEO_PATH = Path(DATA_DIR / "MSRVTT/videos/all")
MSVD_VIDEO_PATH = Path(DATA_DIR / "msvd_videos")


def resolve_video(dataset: str, video_id: str):
    if dataset == "msrvtt":
        return resolve_msrvtt_video(video_id)
    elif dataset == "msvd":
        return resolve_msvd_video(video_id)
    else:
        raise ValueError()


def resolve_msrvtt_video(video_id: str):
    return (MSRVTT_VIDEO_PATH / f"{video_id}.mp4").resolve()


def resolve_msvd_video(video_id: str):
    avi_video_path = (MSVD_VIDEO_PATH / "YouTubeClips" / f"{video_id}.avi").resolve()
    mp4_video_path = (MSVD_VIDEO_PATH / "YouTubeClipsMP4" / f"{video_id}.mp4").resolve()
    if mp4_video_path.exists():
        return str(mp4_video_path)
    else:
        with st.spinner("Converting AVI to MP4 with ffmpeg"):
            subprocess.run(
                f"ffmpeg -i {avi_video_path} {mp4_video_path}", check=True, shell=True
            )

        return mp4_video_path


msrvtt_dataset, msvd_dataset = load_data()

st.header("Dataset Viewer for MSRVTT/MSVD")
st.sidebar.header("Controls")
dataset_name = st.sidebar.selectbox("Dataset", ["msrvtt", "msvd"])
if dataset_name == "msrvtt":
    dataset = msrvtt_dataset
elif dataset_name == "msvd":
    dataset = msvd_dataset
else:
    raise ValueError()

video_id = st.sidebar.selectbox("Video ID", dataset.video_ids)

video_path = resolve_video(dataset_name, video_id)
if dataset_name == "msrvtt":
    video_idx = dataset.vid_to_idx[video_id]
    caption = dataset.captions[video_idx]
elif dataset_name == "msvd":
    video_captions = list(dataset.captions[video_id].keys())
    caption_idx = st.sidebar.selectbox("Caption Index", video_captions)
    if caption_idx in video_captions:
        caption = video_captions[caption_idx]
    else:
        caption = "NO CAPTION"
        st.error(video_captions)
else:
    raise ValueError()

st.text(f"Video ID: {video_id}")
st.text(f"Caption: {caption}")
st.video(f"{video_path}")
