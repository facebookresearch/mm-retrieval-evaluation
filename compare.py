import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path

from mmr.analysis import MsrvttEvaluation, MsvdEvaluation

DATA_DIR = Path(os.environ.get("MMR_DATA", "data"))


class MSRVTT:
    MSRVTT_VIDEO_PATH = DATA_DIR / "MSRVTT/videos/all"
    CSV_PATH = DATA_DIR / "msrvtt_data/MSRVTT_JSFUSION_test.csv"

    def __init__(self, js_fusion_csv: str = CSV_PATH) -> None:
        self.df = pd.read_csv(js_fusion_csv)
        self.idx_to_vid = {}
        self.vid_to_idx = {}
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            self.idx_to_vid[i] = row.video_id
            self.vid_to_idx[row.video_id] = i
        self.captions = self.df["sentence"].values
        self.video_ids = self.df["video_id"].values

    def video_path(self, video_id):
        return os.path.join(self.MSRVTT_VIDEO_PATH, f"{video_id}.mp4")


class ModelPredictions:
    def __init__(self, path: str) -> None:
        # path is directory holding the 2 files
        # predictions_test.json and sim_test.npy
        self.path = path
        self.predictions_test_filename = "predictions_test.json"
        self.sim_test_filename = "sim_test.npy"
        self.caption_to_predictions = {}
        self.caption_to_gold = {}
        self.caption_to_orig_labels = {}
        self.caption_to_new_labels = {}

        self._load()

    def _load(self):
        # with open(os.path.join(self.path, self.predictions_test_filename), 'r') as f:
        #     preds = json.load(f)
        sim_path = Path(self.path) / self.sim_test_filename
        evaluation = MsrvttEvaluation(sim_matrix_path=sim_path, data_dir=DATA_DIR)

        preds = evaluation.score_model(
            crowd_annotations_path=DATA_DIR / "fire_msrvtt_dataset.json"
        )

        for k, v in preds.items():
            self.caption_to_predictions[v["caption"]] = v["preds"]
            self.caption_to_gold[v["caption"]] = v["gold"]
            self.caption_to_orig_labels[v["caption"]] = [
                True if vid == v["gold"] else False for vid in v["preds"]
            ]
            self.caption_to_new_labels[v["caption"]] = [
                True if vid in v["annot_gold"] else False for vid in v["preds"]
            ]


class MSRVTTPredictions:
    def __init__(self) -> None:
        self.data_dir = os.path.join(DATA_DIR / "predictions/msrvtt")
        self.models = ["clip4clip", "collaborative-experts", "ssb"]
        self.predictions = {}

        for m in self.models:
            path = os.path.join(self.data_dir, m)
            self.predictions[m] = ModelPredictions(path)

    def get_prediction_results(self, model: str, caption: str) -> None:
        assert model in self.models, self.models
        if caption not in self.predictions[model].caption_to_predictions:
            return [], [], []
        else:
            return (
                self.predictions[model].caption_to_predictions[caption],
                self.predictions[model].caption_to_orig_labels[caption],
                self.predictions[model].caption_to_new_labels[caption],
            )


@st.experimental_memo
def load_data():
    msrvtt = MSRVTT()
    msrvtt_model_predictions = MSRVTTPredictions()
    return msrvtt, msrvtt_model_predictions


msrvtt_dataset, msrvtt_model_predictions = load_data()

st.header("Model Comparison Viewer for MSRVTT/MSVD")
st.sidebar.header("Controls")

dataset_name = st.sidebar.selectbox("Dataset", ["msrvtt", "msvd"])
if dataset_name == "msrvtt":
    dataset = msrvtt_dataset
# elif dataset_name == "msvd":
# dataset = msvd_dataset
else:
    raise ValueError()


caption = st.sidebar.selectbox("Captions", msrvtt_dataset.captions)
st.markdown("**🏷️: Original Dataset Label, 🔥: FIRE Dataset Label**")
st.markdown(f"**Caption:** {caption}")
OG = "🏷️"
FIRE = "🔥"

clip4clip_col, col_exp_col, ssb_col = st.columns(3)

results_to_show = 5


def label_text(relevance: bool):
    return "✅ Relevant" if relevance else "❌ Irrelevant"


with clip4clip_col:
    st.header("Clip4Clip")
    (
        video_ids,
        orig_labels,
        new_labels,
    ) = msrvtt_model_predictions.get_prediction_results("clip4clip", caption)
    for video_id, orig_label, new_label in zip(
        video_ids[:results_to_show],
        orig_labels[:results_to_show],
        new_labels[:results_to_show],
    ):
        v_path = msrvtt_dataset.video_path(video_id)
        st.video(v_path)
        # We keep the original labels, so new label is either old or new
        st.markdown(f"**{OG} Label**: {label_text(orig_label)}")
        st.markdown(f"**{FIRE} Label**: {label_text(new_label or orig_label)}")

with col_exp_col:
    st.header("Experts")
    (
        video_ids,
        orig_labels,
        new_labels,
    ) = msrvtt_model_predictions.get_prediction_results(
        "collaborative-experts", caption
    )
    for video_id, orig_label, new_label in zip(
        video_ids[:results_to_show],
        orig_labels[:results_to_show],
        new_labels[:results_to_show],
    ):
        v_path = msrvtt_dataset.video_path(video_id)
        st.video(v_path)
        st.markdown(f"**{OG} Label**: {label_text(orig_label)}")
        st.markdown(f"**{FIRE} Label**: {label_text(new_label or orig_label)}")

with ssb_col:
    st.header("SSB")
    (
        video_ids,
        orig_labels,
        new_labels,
    ) = msrvtt_model_predictions.get_prediction_results("ssb", caption)
    for video_id, orig_label, new_label in zip(
        video_ids[:results_to_show],
        orig_labels[:results_to_show],
        new_labels[:results_to_show],
    ):
        v_path = msrvtt_dataset.video_path(video_id)
        st.video(v_path)
        st.markdown(f"**{OG} Label**: {label_text(orig_label)}")
        st.markdown(f"**{FIRE} Label**: {label_text(new_label or orig_label)}")
