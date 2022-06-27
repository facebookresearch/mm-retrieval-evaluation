import pandas as pd
import os
import streamlit as st
from pathlib import Path
from pedroai.io import read_json
from mmr.matcher import (
    CharacterNGramMatcher,
    EditDistanceMatcher,
    Similarity,
    WordNGramMatcher,
)
from mmr.tasks import (
    AllAutomaticMsrvttEvals,
    AllAutomaticMsrvttMatchingLabels,
    PERCENTILE,
    HIGH_SCORE,
    CHAR,
    WORD,
    EDIT,
)
from mmr.datasets import Annotation, FireDataset


DATA_DIR = Path(os.environ.get("MMR_DATA", "data"))
MATCHER_TYPES = [CHAR, WORD, EDIT]
THRESHOLD_TYPES = [PERCENTILE, HIGH_SCORE]


@st.experimental_memo
def load_data():
    predictions = {}
    base_path = DATA_DIR / "predictions/msrvtt"
    models = ["clip4clip", "collaborative-experts", "ssb"]
    for m in models:
        predictions[m] = {
            int(k): v
            for k, v in read_json(base_path / m / "predictions_test.json").items()
        }

    msrvtt_fire = FireDataset.parse_file(DATA_DIR / "fire_msrvtt_dataset.json")
    dataset = pd.read_csv(DATA_DIR / "msrvtt_data/MSRVTT_JSFUSION_test.csv")
    corpus = []
    for r in dataset.itertuples():
        corpus.append(r.sentence)
    return corpus, predictions, msrvtt_fire


st.header("Compare Text Matchers for Automatic Labeling")
st.sidebar.header("Configuration")
matcher_type = st.sidebar.selectbox("Matcher", MATCHER_TYPES)
run_matcher = st.sidebar.button("Run Matcher")
corpus, predictions, msrvtt_fire = load_data()


@st.experimental_memo
def compute_matches(matcher_type):
    if matcher_type == CHAR:
        matcher = CharacterNGramMatcher(corpus)
    elif matcher_type == WORD:
        matcher = WordNGramMatcher(corpus)
    elif matcher_type == EDIT:
        matcher = EditDistanceMatcher()
    else:
        raise ValueError("Invalid matcher type")
    _, candidates = matcher.similarity(corpus)
    candidates = sorted(candidates, key=lambda c: c.score, reverse=matcher_type != EDIT)
    rows = []
    for c in candidates:
        rows.append({"text_a": c.first_text, "text_b": c.second_text, "score": c.score})
    df = pd.DataFrame(rows)
    stats = df.quantile([0.5, 0.8, 0.9, 0.95, 0.99])
    return df, stats


if run_matcher:
    df, stats = compute_matches(matcher_type)
else:
    df = pd.DataFrame()
    stats = pd.DataFrame()

st.header("Percentile Stats")
st.table(stats)
st.header("Text Matches")
st.table(df.head(200))
