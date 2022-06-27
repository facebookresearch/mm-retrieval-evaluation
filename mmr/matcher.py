"""
Implement the automatic text matching methods
"""
from pathlib import Path
import pandas as pd
from rich.console import Console
import torch
import altair as alt
import abc
import numpy as np
from typing import Dict, List, Tuple
import pydantic
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from transformers import BertModel, BertTokenizer
from functional import pseq
from mmr.analysis import MsrvttEvaluation
from pedroai.altair import save_chart

from mmr.datasets import Annotation, MsrVttDataset
from mmr.io import read_json


GoldAnnotationDataset = Dict[Tuple[str, str], Annotation]
BERT_MODEL = "bert-base-uncased"
console = Console()


class Similarity(pydantic.BaseModel):
    first_idx: int
    first_text: str
    second_idx: int
    second_text: str
    # Let's always assume more positive is more similar
    score: float


class Matcher(abc.ABC):
    @abc.abstractmethod
    def similarity(self, texts: List[str]) -> Tuple[np.ndarray, List[Similarity]]:
        pass


def similarity_matrix_to_pairs(
    texts: List[str], distances: np.ndarray
) -> List[Similarity]:
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Non-square matrix dimensions: {distances.shape}")
    n_texts = distances.shape[0]
    rows = []
    for i in range(n_texts):
        for j in range(n_texts):
            dist = distances[i, j]
            if i != j and i > j:
                rows.append(
                    Similarity(
                        first_idx=i,
                        second_idx=j,
                        first_text=texts[i],
                        second_text=texts[j],
                        score=dist,
                    )
                )
    return rows


def par_compute(texts: List[str]):
    n_texts = len(texts)

    def unit(i: int):
        distances = []
        for j in range(n_texts):
            if i > j:
                distances.append((i, j, Levenshtein.distance(texts[i], texts[j])))
        return distances

    return unit


class EditDistanceMatcher(Matcher):
    def similarity(self, texts: List[str]) -> Tuple[np.ndarray, List[Similarity]]:
        console.log("Preparing texts")
        n_texts = len(texts)
        distances = np.zeros((n_texts, n_texts))

        console.log("Computing distances in parallel")
        dist_triples = pseq(range(n_texts)).flat_map(par_compute(texts)).list()
        for i, j, dist in dist_triples:
            distances[i, j] = dist
            distances[j, i] = dist

        # for i in range(n_texts):
        #     for j in range(n_texts):
        #         if i > j:
        #             dist = Levenshtein.distance(texts[i], texts[j])
        #             distances[i, j] = dist
        #             distances[j, i] = dist
        rows = similarity_matrix_to_pairs(texts, distances)
        return distances, rows


class TfidfMatcher(Matcher):
    def __init__(
        self,
        corpus: List[str],
        *,
        ngram_range: Tuple[int, int],
        analyzer: str,
        lowercase: bool = True,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.x_data = np.array(corpus)
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        console.log(f"Indexing corpus with n_texts={len(corpus)}")
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range, lowercase=lowercase, analyzer=analyzer
        ).fit(self.x_data)
        self.matrix = self.tfidf.transform(self.x_data)

    def search(self, text: str, n=10):
        reps = self.tfidf.transform([text])
        result = self.matrix.dot(reps.T).T
        top_indices = np.argsort(-result.toarray()[0])
        return self.data.iloc[top_indices[:n]]

    def similarity(self, texts: List[str]) -> Tuple[np.ndarray, List[Similarity]]:
        console.log("Processing Input Text")
        reps = self.tfidf.transform(texts)
        console.log("Computing similarity matrix")
        distances = self.matrix.dot(reps.T).T
        console.log("Converting to pairs")
        rows = similarity_matrix_to_pairs(texts, distances)
        return distances, rows


class CharacterNGramMatcher(TfidfMatcher):
    def __init__(
        self, corpus: List[str], ngram_range: Tuple[int, int] = (1, 5)
    ) -> None:
        super().__init__(corpus, ngram_range=ngram_range, analyzer="char_wb")


class WordNGramMatcher(TfidfMatcher):
    def __init__(
        self, corpus: List[str], ngram_range: Tuple[int, int] = (1, 3)
    ) -> None:
        super().__init__(corpus, ngram_range=ngram_range, analyzer="word")


class EmbeddingMatcher(Matcher):
    def __init__(self, corpus: List[str]) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.model = BertModel.from_pretrained(BERT_MODEL)
        input_ids = tokenizer
        with torch.no_grad():
            last_hiddens = self.model()


def threshold_percentile(
    candidates: List[Similarity], threshold: float, higher_is_better: bool
) -> List[Similarity]:
    """
    Note, threshold should be between 0-100
    """
    if higher_is_better:
        scores = [s.score for s in candidates]
        cutoff = np.percentile(scores, threshold)
        return [s for s in candidates if s.score >= cutoff]
    else:
        scores = [s.score for s in candidates]
        cutoff = np.percentile(scores, 100 - threshold)
        return [s for s in candidates if s.score <= cutoff]


def threshold_high_score(
    candidates: List[Similarity], threshold: float, higher_is_better: bool
) -> List[Similarity]:
    if higher_is_better:
        return [c for c in candidates if c.score >= threshold]
    else:
        return [c for c in candidates if c.score <= threshold]


PERCENTILE = "PERCENTILE"
HIGH_SCORE = "HIGH_SCORE"
CHAR = "char"
WORD = "word"
EDIT = "edit"
EMB = "emb"


def create_msrvtt_automatic_annotations(
    *,
    matcher_type: str,
    threshold_type: str,
    higher_is_better: bool,
    threshold: float,
) -> Tuple[Dict[Tuple[str, str], Annotation], List[Similarity]]:
    """
    Generate automatic labels between caption-videos by first looking at similar captions.
    If two captions are similar to each other, then consider their videos matches to each other.
    Do this using:
    1) The supplied text matching method
    2) Using the thresholding method to cut down
    3) Using the threshold value, which can be a threshold on the score or a percentile typically
    """
    console.log("Creating MSRVTT Automatic Annotations with:")
    console.log(locals())
    dataset = pd.read_csv("data/msrvtt_data/MSRVTT_JSFUSION_test.csv")
    console.log(f"Data has N rows = {len(dataset)}")
    corpus = []
    caption_to_relevant_videos = {}
    for r in dataset.itertuples():
        corpus.append(r.sentence)
        if r.sentence not in caption_to_relevant_videos:
            caption_to_relevant_videos[r.sentence] = set()
        caption_to_relevant_videos[r.sentence].add(r.video_id)

    if matcher_type == CHAR:
        matcher = CharacterNGramMatcher(corpus)
    elif matcher_type == WORD:
        matcher = WordNGramMatcher(corpus)
    elif matcher_type == EDIT:
        matcher = EditDistanceMatcher()
    elif matcher_type == EMB:
        matcher = EmbeddingMatcher()
    else:
        raise ValueError("Invalid matcher type")

    _, candidates = matcher.similarity(corpus)
    if threshold_type == PERCENTILE:
        selected_labels = threshold_percentile(candidates, threshold, higher_is_better)
    elif threshold_type == HIGH_SCORE:
        selected_labels = threshold_high_score(candidates, threshold, higher_is_better)
    else:
        raise ValueError("Invalid threshold type")

    annotations: Dict[Tuple[str, str], Annotation] = {}
    for text_pair in selected_labels:
        for relevant_video in caption_to_relevant_videos[text_pair.second_text]:
            annotations[(text_pair.first_text, relevant_video)] = Annotation(
                caption=text_pair.first_text,
                video_id=relevant_video,
                relevant=True,
            )
        for relevant_video in caption_to_relevant_videos[text_pair.first_text]:
            annotations[(text_pair.second_text, relevant_video)] = Annotation(
                caption=text_pair.second_text,
                video_id=relevant_video,
                relevant=True,
            )
    return annotations, candidates


def train_test_overlap_analysis(data_path: str, out_dir: str, k: int = 10):
    data_path = Path(data_path)
    train_df = pd.read_csv(data_path / "msrvtt_data/MSRVTT_train.9k.csv")
    test_df = pd.read_csv(data_path / "msrvtt_data/MSRVTT_JSFUSION_test.csv")
    msrvtt = MsrVttDataset.from_file()
    train_videos = set(train_df.video_id.values)
    test_videos = set(test_df.video_id.values)
    train_pairs = []
    test_pairs = []
    failed = []
    for c in msrvtt.sentences:
        if c.video_id in train_videos:
            train_pairs.append(c)
        elif c.video_id in test_videos:
            test_pairs.append(c)
        else:
            failed.append(c)
    if len(failed) > 0:
        raise ValueError()
    train_captions = [c.caption for c in train_pairs]
    test_captions = [r.sentence for r in test_df.itertuples()]
    matcher = CharacterNGramMatcher(test_captions)
    reps = matcher.tfidf.transform(train_captions)
    distances = matcher.matrix.dot(reps.T)
    top_captions = np.argsort(-distances.toarray())

    clip_eval = MsrvttEvaluation(
        sim_matrix_path=data_path / "predictions/msrvtt/clip4clip/sim_test.npy",
        data_dir=data_path,
    )
    clip_cap_to_preds = clip_eval.score_model(
        crowd_annotations_path=data_path / "fire_msrvtt_dataset.json"
    )
    clip_orig_correct = {}
    clip_fire_correct = {}
    for preds in clip_cap_to_preds.values():
        clip_orig_correct[preds["caption"]] = preds["metrics"][f"C@{k}"]
        clip_fire_correct[preds["caption"]] = preds["metrics_annotations"][f"C@{k}"]

    tt_eval = MsrvttEvaluation(
        sim_matrix_path=data_path
        / "predictions/msrvtt/collaborative-experts/sim_test.npy",
        data_dir=data_path,
    )
    tt_cap_to_preds = tt_eval.score_model(
        crowd_annotations_path=data_path / "fire_msrvtt_dataset.json"
    )
    tt_orig_correct = {}
    tt_fire_correct = {}
    for preds in tt_cap_to_preds.values():
        tt_orig_correct[preds["caption"]] = preds["metrics"][f"C@{k}"]
        tt_fire_correct[preds["caption"]] = preds["metrics_annotations"][f"C@{k}"]

    rows = []
    for i in range(len(test_captions)):
        scores = []
        caption = test_captions[i]
        for j in range(k):
            idx = top_captions[i, j]
            scores.append(distances[i, idx])

        if clip_orig_correct[caption] == 1 and tt_orig_correct[caption] == 1:
            orig_group = "Both"
        elif clip_orig_correct[caption] == 1 and tt_orig_correct[caption] == 0:
            orig_group = "CLIP"
        elif clip_orig_correct[caption] == 0 and tt_orig_correct[caption] == 1:
            orig_group = "TT"
        elif clip_orig_correct[caption] == 0 and tt_orig_correct[caption] == 0:
            orig_group = "None"
        else:
            raise ValueError()

        if clip_fire_correct[caption] == 1 and tt_fire_correct[caption] == 1:
            fire_group = "Both"
        elif clip_fire_correct[caption] == 1 and tt_fire_correct[caption] == 0:
            fire_group = "CLIP"
        elif clip_fire_correct[caption] == 0 and tt_fire_correct[caption] == 1:
            fire_group = "TT"
        elif clip_fire_correct[caption] == 0 and tt_fire_correct[caption] == 0:
            fire_group = "None"
        else:
            raise ValueError()

        rows.append(
            {
                "test_caption": caption,
                "mean_score": np.mean(scores),
                "group": orig_group,
                "annotations": "Original",
            }
        )
        rows.append(
            {
                "test_caption": caption,
                "mean_score": np.mean(scores),
                "group": fire_group,
                "annotations": "FIRE",
            }
        )

    sim_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    sim_df.to_feather(out_dir / f"text_similarity_{k}.feather")
