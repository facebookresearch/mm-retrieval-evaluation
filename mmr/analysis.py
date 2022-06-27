from typing import Set, List, Dict, Tuple, Optional
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from mmr.datasets import (
    convert_crowdsource_to_gold_format,
    parse_annotations,
    GoldAnnotation,
)


def compute_precision(gold: Set[str], ranked_list: List[str], ranks=(1, 5, 10)):
    metrics = {}
    for n in ranks:
        metrics[f"P@{n}"] = len(set(ranked_list[:n]) & gold) / n
    return metrics


def compute_recall(gold: Set[str], ranked_list: List[str], ranks=(1, 5, 10)):
    metrics = {}
    for n in ranks:
        metrics[f"R@{n}"] = len(set(ranked_list[:n]) & gold) / max(1, len(gold))
    return metrics


def compute_correct_at_k(gold: Set[str], ranked_list: List[str], ranks=(1, 5, 10)):
    metrics = {}
    for n in ranks:
        metrics[f"C@{n}"] = int(len(gold & set(ranked_list[:n])) > 0)
    return metrics


def compute_average_precision(gold: Set[str], ranked_list: List[str]):
    precisions = []
    hits = []
    for i in range(len(ranked_list)):
        gold_at_rank = ranked_list[i] in gold
        if gold_at_rank:
            hits.append(1)
            precisions.append(np.mean(hits))
        else:
            hits.append(0)
    if len(precisions) == 0:
        return {"AP": 0.0}
    else:
        return {"AP": np.mean(precisions)}


def compute_metrics(gold_set: Set[str], top_videos: List[str]):
    precision = compute_precision(gold_set, top_videos)
    recall = compute_recall(gold_set, top_videos)
    ap = compute_average_precision(gold_set, top_videos)
    correct_at_k = compute_correct_at_k(gold_set, top_videos)
    return {**precision, **recall, **ap, **correct_at_k}


class MsrvttEvaluation:
    def __init__(
        self, *, sim_matrix_path: str, data_dir: str, max_preds: int = 20
    ) -> None:
        self.sim_matrix_path = sim_matrix_path
        self.data_dir = Path(data_dir)
        self.max_preds = max_preds

    def sim_matrix_to_readable(self):
        df = pd.read_csv(self.data_dir / "msrvtt_data/MSRVTT_JSFUSION_test.csv")
        preds = np.load(self.sim_matrix_path)
        caption_id_to_predictions = {}
        for i_cap in range(preds.shape[0]):
            top_videos = [
                df.iloc[i].video_id for i in np.argsort(-preds[i_cap])[: self.max_preds]
            ]
            caption = df.iloc[i_cap].sentence
            gold = df.iloc[i_cap].video_id
            gold_set = set([gold])
            caption_id_to_predictions[i_cap] = {
                "gold": gold,
                "caption": caption,
                "caption_idx": i_cap,
                "preds": top_videos,
                "metrics": compute_metrics(gold_set, top_videos),
            }
        return caption_id_to_predictions

    def score_model(
        self,
        analysis_annotations_path: str = None,
        crowd_annotations_path: str = None,
        crowd_annotations: Optional[Dict] = None,
    ):
        df = pd.read_csv(self.data_dir / "msrvtt_data/MSRVTT_JSFUSION_test.csv")
        # all_captions = set(df["sentence"])
        preds = np.load(self.sim_matrix_path)
        if crowd_annotations is not None:
            annotations = crowd_annotations
        elif analysis_annotations_path is not None:
            annotations = parse_annotations(analysis_annotations_path)
        elif crowd_annotations_path is not None:
            annotations = convert_crowdsource_to_gold_format(crowd_annotations_path)
        else:
            annotations = {}
        captions_to_annotations = {}
        for a in annotations.values():
            if a.caption not in captions_to_annotations:
                captions_to_annotations[a.caption] = []
            captions_to_annotations[a.caption].append(a)
        caption_id_to_predictions = {}
        for i_cap in range(preds.shape[0]):
            top_videos = [
                df.iloc[i].video_id for i in np.argsort(-preds[i_cap])[: self.max_preds]
            ]
            caption = df.iloc[i_cap].sentence
            gold = df.iloc[i_cap].video_id
            gold_set = set([gold])
            if caption in captions_to_annotations:
                annot_gold = {
                    a.video_id for a in captions_to_annotations[caption] if a.relevant
                }
                all_annot = {a.video_id for a in captions_to_annotations[caption]}
            else:
                annot_gold = set()
                all_annot = set()
            # Hard code 10 since this is what we annotated
            p_annotated_preds = len(
                (gold_set | all_annot) & set(top_videos[:10])
            ) / len(set(top_videos[:10]))
            caption_id_to_predictions[i_cap] = {
                "idx": i_cap,
                "gold": gold,
                "annot_gold": list(annot_gold),
                "caption": caption,
                "preds": top_videos,
                "metrics": compute_metrics(gold_set, top_videos),
                "metrics_annotations": compute_metrics(
                    gold_set | annot_gold, top_videos
                ),
                "has_gold_annotation": len(annot_gold) > 0,
                "p_annotated_preds": p_annotated_preds,
            }
        return caption_id_to_predictions

    def get_gold_annotations(self) -> Dict[Tuple[str, str], GoldAnnotation]:
        df = pd.read_csv(self.data_dir / "msrvtt_data/MSRVTT_JSFUSION_test.csv")
        gold = {}
        for i in range(len(df)):
            row = df.iloc[i]
            gold[(row.sentence, row.video_id)] = GoldAnnotation(
                caption=row.sentence, video_id=row.video_id, idx=i
            )
        return gold


class MsvdEvaluation:
    def __init__(
        self, *, sim_matrix_path: str, data_dir: str, max_preds: int = 20
    ) -> None:
        self.max_preds = max_preds
        self.sim_matrix_path = sim_matrix_path
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "msvd_data/test_list.txt") as f:
            video_ids = [r.strip() for r in f]
        self.idx_to_vid = {idx: vid for idx, vid in enumerate(video_ids)}
        self.vid_to_idx = {vid: idx for idx, vid in self.idx_to_vid.items()}
        with open("data/msvd_data/raw-captions.pkl", "rb") as f:
            msvd_captions = pickle.load(f)
        caption_map = {}
        for video_id, captions in msvd_captions.items():
            for idx, c in enumerate(captions):
                text = " ".join(c)
                caption_map[(video_id, idx)] = text
        self.captions = caption_map

    def sim_matrix_to_readable(self, ce_meta_path: Optional[str] = None):
        if ce_meta_path is not None:
            return self._sim_matrix_to_readable_ce(ce_meta_path)
        msvd_sim = np.load(self.sim_matrix_path)
        preds = {}
        for i in range(msvd_sim.shape[0]):
            video_id = self.idx_to_vid[i]
            if video_id not in preds:
                preds[video_id] = {}
            for j in range(msvd_sim.shape[1]):
                per_caption_preds = msvd_sim[i, j, :]
                if np.isinf(per_caption_preds).sum() > 0:
                    continue
                else:
                    preds[video_id][j] = per_caption_preds
        msvd_ranked_videos = {}
        for video_id, caption_idx_to_scores in preds.items():
            msvd_ranked_videos[video_id] = {}
            for cap_idx, scores in caption_idx_to_scores.items():
                text = self.captions[(video_id, cap_idx)]
                gold = [video_id]
                preds = [
                    self.idx_to_vid[idx]
                    for idx in np.argsort(-scores)[: self.max_preds]
                ]
                msvd_ranked_videos[video_id][cap_idx] = {
                    "gold": gold,
                    "preds": preds,
                    "caption": text,
                    "caption_idx": cap_idx,
                    "metrics": compute_metrics(set(gold), preds),
                }
        return msvd_ranked_videos

    def _sim_matrix_to_readable_ce(self, ce_meta_path: str):
        unshaped_msvd_sims = np.load(self.sim_matrix_path)
        msvd_sims = unshaped_msvd_sims.reshape(670, -1, 670)
        with open(ce_meta_path, "rb") as f:
            meta = pickle.load(f)
        # raw_captions = meta["raw_captions"]
        query_masks = meta["query_masks"].astype(int)
        n_videos = msvd_sims.shape[0]
        max_captions = msvd_sims.shape[1]
        all_preds = {}
        for i in range(n_videos):
            video_id = self.idx_to_vid[i]
            if video_id not in all_preds:
                all_preds[video_id] = {}
            for j in range(max_captions):
                is_real_caption = query_masks[i, j]
                if is_real_caption == 1:
                    per_caption_scores = msvd_sims[i, j, :]
                    per_caption_preds = np.argsort(-per_caption_scores)
                    all_preds[video_id][j] = per_caption_preds
                elif is_real_caption == 0:
                    continue
                else:
                    raise ValueError()

        msvd_ranked_videos = {}
        for video_id, caption_idx_to_scores in all_preds.items():
            msvd_ranked_videos[video_id] = {}
            for cap_idx, sorted_preds in caption_idx_to_scores.items():
                text = self.captions[(video_id, cap_idx)]
                gold = [video_id]
                preds = [self.idx_to_vid[idx] for idx in sorted_preds[: self.max_preds]]
                msvd_ranked_videos[video_id][cap_idx] = {
                    "gold": gold,
                    "preds": preds,
                    "caption": text,
                    "caption_idx": cap_idx,
                    "metrics": compute_metrics(set(gold), preds),
                }
        return msvd_ranked_videos

    def score_model(
        self,
        crowd_annotations_path: str = None,
        ce_meta_path: str = None,
        crowd_annotations: Optional[Dict] = None,
    ):
        if crowd_annotations is not None:
            annotations = crowd_annotations
        elif crowd_annotations_path is None:
            annotations = {}
        else:
            annotations = convert_crowdsource_to_gold_format(crowd_annotations_path)
            captions_to_annotations = {}
            for a in annotations.values():
                if a.caption not in captions_to_annotations:
                    captions_to_annotations[a.caption] = []
                captions_to_annotations[a.caption].append(a)

        ranked_preds = self.sim_matrix_to_readable(ce_meta_path=ce_meta_path)

        for video_id, cap_to_result in ranked_preds.items():
            for cap_idx, result in cap_to_result.items():
                caption = self.captions[(video_id, cap_idx)]
                if caption in captions_to_annotations:
                    annot_gold = {
                        a.video_id
                        for a in captions_to_annotations[caption]
                        if a.relevant
                    }
                    all_annot = {a.video_id for a in captions_to_annotations[caption]}
                else:
                    annot_gold = set()
                    all_annot = set()
                gold_set = set(result["gold"])
                top_videos = result["preds"][:10]
                p_annotated_preds = len((gold_set | all_annot) & set(top_videos)) / len(
                    set(top_videos)
                )
                result["annot_gold"] = list(annot_gold)
                result["metrics_annotations"] = compute_metrics(
                    gold_set | annot_gold, top_videos
                )
                result["has_gold_annotations"] = len(annot_gold) > 0
                result["p_annotated_preds"] = p_annotated_preds
        return ranked_preds
