from typing import List, Type, TypeVar, Dict, Tuple
import pandas as pd
import json
from enum import Enum
from pydantic import BaseModel
from mmr.config import MSRVTT_JSON
from mmr.io import read_json


T = TypeVar("T", bound="MsrVttDataset")


class LabelEnum(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"

    def __str__(self):
        return self.name


class ModelEnum(str, Enum):
    CLIP4CLIP = "CLIP4CLIP"
    SSB = "SSB"
    CE = "CE"

    def __str__(self):
        return self.name


class DatasetEnum(str, Enum):
    MSVD = "MSVD"
    MSRVTT = "MSRVTT"

    def __str__(self):
        return self.name


class FireAnnotation(BaseModel):
    annotator_ids: List[int]
    label: LabelEnum
    video_id: str
    query: str
    annotator_labels: List[LabelEnum]
    response_ids: List[int]
    queue_ids: List[int]
    models: List[ModelEnum]
    dataset: DatasetEnum


class FireDisagreement(BaseModel):
    annotator_ids: List[int]
    video_id: str
    query: str
    annotator_labels: List[LabelEnum]
    response_ids: List[int]
    queue_ids: List[int]
    models: List[ModelEnum]
    dataset: DatasetEnum


class FireDataset(BaseModel):
    annotations: List[FireAnnotation]
    disagreements: List[FireDisagreement]
    source_ds: str
    source_data_md5: str
    source_annotator_map_md5: str
    created_ds: str
    email: str
    creator: str
    website: str
    license: str
    datasets: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for a in self.annotations:
            entry = a.dict()
            entry["agreement"] = True
            rows.append(entry)

        for d in self.disagreements:
            entry = d.dict()
            entry["agreement"] = False
            entry["label"] = None
        return pd.DataFrame(rows)


class MsrVttCaption(BaseModel):
    caption: str
    video_id: str
    sen_id: int


class MsrVttVideo(BaseModel):
    category: int
    url: str
    video_id: str
    start_time: float
    end_time: float
    split: str
    id_: int


class MsrVttInfo(BaseModel):
    contributor: str
    data_created: str
    version: str
    description: str
    year: str


class MsrVttDataset(BaseModel):
    info: MsrVttInfo
    videos: List[MsrVttVideo]
    sentences: List[MsrVttCaption]

    @classmethod
    def from_file(cls: Type[T], file: str = MSRVTT_JSON) -> T:
        with open(file) as f:
            data = json.load(f)
            return MsrVttDataset(
                info=data["info"],
                sentences=data["sentences"],
                videos=[
                    MsrVttVideo(
                        category=v["category"],
                        url=v["url"],
                        video_id=v["video_id"],
                        start_time=v["start time"],
                        end_time=v["end time"],
                        split=v["split"],
                        id_=v["id"],
                    )
                    for v in data["videos"]
                ],
            )


class Annotation(BaseModel):
    video_id: str
    caption: str
    relevant: bool

    class Config:
        frozen = True


class GoldAnnotation(BaseModel):
    video_id: str
    caption: str
    idx: int

    class Config:
        frozen = True


def parse_annotations(file_path: str) -> Dict[Tuple[str, str], Annotation]:
    annotations_dict = read_json(file_path)
    annotations = {}
    for key, relevant in annotations_dict.items():
        video_id, caption = key.split(":")
        key = (caption, video_id)
        if key in annotations:
            raise ValueError(f"Duplicate annotations for: {key}")

        annotations[key] = Annotation(
            video_id=video_id,
            caption=caption,
            relevant=relevant,
        )
    return annotations


def convert_crowdsource_to_gold_format(
    file_path: str,
) -> Dict[Tuple[str, str], Annotation]:
    mmr_dataset = FireDataset.parse_file(file_path)
    annotations: Dict[Tuple[str, str], Annotation] = {}
    for row in mmr_dataset.annotations:
        key = (row.query, row.video_id)
        if row.label == LabelEnum.RELEVANT:
            relevant = True
        elif row.label == LabelEnum.IRRELEVANT:
            relevant = False
        else:
            raise ValueError()
        annotations[key] = Annotation(
            video_id=row.video_id, caption=row.query, relevant=relevant
        )
    return annotations
