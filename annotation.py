import json
import os
import random
import pandas as pd
from pathlib import Path
import streamlit as st
import mmr.io
from mmr.io import read_json


DATA_DIR = Path(os.environ.get("MMR_DATA", "data"))


@st.experimental_memo
def load_preds():
    preds = read_json(DATA_DIR / "msrvtt_preds.json")
    return list(preds.keys()), preds


@st.experimental_memo
def load_captions_to_videos():
    return read_json(DATA_DIR / "caption_to_videos.json")


@st.experimental_memo
def load_video_to_url():
    return read_json(DATA_DIR / "video_to_url.json")


def write_color_text(context, color: str, text: str):
    context.markdown(
        f'<p style="background-color:{color}">{text}</p>', unsafe_allow_html=True
    )


def annotation_key(*, video_id: str, caption: str):
    return f"{video_id}:{caption}"


def annotate_callback(caption: str, video_id: str, relevance: bool):
    st.session_state["annotations"][
        annotation_key(video_id=video_id, caption=caption)
    ] = relevance


def delete_callback(caption: str, video_id: str):
    del st.session_state["annotations"][
        annotation_key(video_id=video_id, caption=caption)
    ]


def write_row(
    *,
    row_number: int,
    caption: str,
    video_url: str,
    correct: str,
    video_id: str,
    annotatable: bool,
):
    left, right = st.columns([1, 3])
    left.write(f"Rank: {row_number} ID: {video_id}")
    left.write(f"Correct: {correct}")
    if annotatable:
        left.button(
            "Mark Relevant",
            key=f"mark_relevant:{caption}:{video_id}:{row_number}",
            on_click=annotate_callback,
            args=(caption, video_id, True),
        )
        left.button(
            "Mark Irrelevant",
            key=f"mark_irrelevant:{caption}:{video_id}:{row_number}",
            on_click=annotate_callback,
            args=(caption, video_id, False),
        )
        key = annotation_key(video_id=video_id, caption=caption)
        if key in st.session_state["annotations"]:
            left.button(
                "Delete Annotation",
                key=f"delete:{caption}:{video_id}:{row_number}",
                on_click=delete_callback,
                args=(caption, video_id),
            )
    right.video(video_url)


def visualize_pred(caption: str, max_n: int = 10):
    prediction = preds[caption]
    gold_video = prediction["gold"]
    write_row(
        row_number=-1,
        caption=caption,
        video_url=video_to_url[gold_video],
        correct="gold",
        video_id=gold_video,
        annotatable=True,
    )
    for idx, video_id in enumerate(prediction["preds"][:max_n]):
        url = video_to_url[video_id]
        correct = video_id == prediction["gold"]
        write_row(
            row_number=idx,
            caption=caption,
            video_url=url,
            correct=str(correct),
            video_id=video_id,
            annotatable=True,
        )


captions, preds = load_preds()
captions_to_videos = load_captions_to_videos()
video_to_url = load_video_to_url()
if "annotations" not in st.session_state:
    if Path(DATA_DIR / "analysis_par_annotations.json").exists():
        st.info("Loaded existing annotations from: data/analysis_par_annotations.json")
        st.session_state["annotations"] = read_json(
            DATA_DIR / "analysis_par_annotations.json"
        )
    else:
        st.session_state["annotations"] = {}


st.header("MSR VTT False Negative Analysis")
st.sidebar.header("Annotations")
save_location = st.sidebar.text_input(
    "Save Annotations To: ", DATA_DIR / "analysis_par_annotations.json"
)


def save_callback():
    with open(save_location, "w") as f:
        json.dump(st.session_state["annotations"], f)


def clear_callback():
    st.session_state["annotations"] = {}


st.sidebar.button("Save Annotations", on_click=save_callback)
st.sidebar.button("Clear Annotations", on_click=clear_callback)


def display_annotations():
    annotations = st.session_state["annotations"]
    rows = []
    for key, label in annotations.items():
        video_id, caption = key.split(":")
        rows.append({"video_id": video_id, "caption": caption, "relevant": label})
    df = pd.DataFrame(rows, columns=["caption", "video_id", "relevant"])
    st.sidebar.subheader("Annotation Stats")
    st.sidebar.write(f"N Annotations: {len(df)}")
    st.sidebar.write(f"N Unique Captions: {len(df['caption'].unique())}")
    st.sidebar.write(f"N Unique Videos: {len(df['caption'].unique())}")
    st.sidebar.write(
        f"N Relevant: {df['relevant'].sum()} N Irrelevant: {(df['relevant'] == False).sum()}"
    )
    st.sidebar.table(df)


display_annotations()

if "selected_caption_idx" not in st.session_state:
    st.session_state["selected_caption_idx"] = 0

if st.button("Random Caption"):
    st.session_state["selected_caption_idx"] = random.randint(0, len(captions) - 1)

if st.button("Next Shortest Caption"):
    annotations = st.session_state["annotations"]
    annotated_captions = set()
    for key in annotations.keys():
        _, caption = key.split(":")
        annotated_captions.add(caption)
    sorted_captions = sorted(captions, key=len)
    captions_to_idx = {}
    for i, caption in enumerate(captions):
        captions_to_idx[caption] = i
    for caption in sorted_captions:
        if caption not in annotated_captions:
            st.session_state["selected_caption_idx"] = captions_to_idx[caption]
            break
    else:
        st.warning("Could not find shortest caption, NOP")

selected_caption = st.selectbox(
    "Choose Caption", captions, index=st.session_state["selected_caption_idx"]
)

visualize_pred(selected_caption)
