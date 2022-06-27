import pickle
from typing import List
import datetime
from pathlib import Path
import json
import luigi
import pandas as pd
import altair as alt
from rich.console import Console
from scipy import stats
from pedroai.io import safe_file
from mmr.analysis import MsrvttEvaluation, MsvdEvaluation
from mmr.io import md5sum, read_json, write_json
from mmr.datasets import (
    FireAnnotation,
    FireDisagreement,
    FireDataset,
    ModelEnum,
    DatasetEnum,
)
from mmr.matcher import (
    PERCENTILE,
    HIGH_SCORE,
    CHAR,
    WORD,
    EDIT,
    create_msrvtt_automatic_annotations,
    train_test_overlap_analysis,
)

SSB = "ssb"
COLLAB_EXPERTS = "collaborative-experts"
CLIP4CLIP = "clip4clip"
HIT = "hit"
MODELS = [CLIP4CLIP, COLLAB_EXPERTS, SSB, HIT]

MSRVTT = "msrvtt"
MSVD = "msvd"
DATASETS = [MSRVTT, MSVD]

MODEL_PREDICTIONS = [
    (MSRVTT, CLIP4CLIP),
    (MSRVTT, COLLAB_EXPERTS),
    (MSRVTT, SSB),
    (MSVD, CLIP4CLIP),
    (MSVD, COLLAB_EXPERTS),
]
console = Console()


class MsrvttData(luigi.ExternalTask):
    def output(self):
        return [
            luigi.LocalTarget("data/msrvtt_data/MSRVTT_JSFUSION_test.csv"),
        ]


class MsvdData(luigi.ExternalTask):
    def output(self):
        return [
            luigi.LocalTarget("data/msvd_data/test_list.txt"),
            luigi.LocalTarget("data/msvd_data/raw-captions.pkl"),
        ]


class SimilarityMatrix(luigi.ExternalTask):
    model = luigi.ChoiceParameter(choices=MODELS)
    dataset = luigi.ChoiceParameter(choices=DATASETS)

    def output(self):
        base_path = Path("data/predictions")
        return luigi.LocalTarget(base_path / self.dataset / self.model / "sim_test.npy")


class MsvttSimToPreds(luigi.Task):
    model = luigi.ChoiceParameter(choices=MODELS)

    def requires(self):
        yield MsrvttData()
        yield SimilarityMatrix(model=self.model, dataset="msrvtt")

    @property
    def _output_path(self):
        base_path = Path("data/predictions")
        return base_path / "msrvtt" / self.model / "predictions_test.json"

    def run(self):
        sim_matrix_path = Path("data/predictions/msrvtt") / self.model / "sim_test.npy"
        caption_id_to_preds = MsrvttEvaluation(
            sim_matrix_path=sim_matrix_path, data_dir="data"
        ).sim_matrix_to_readable()
        with open(self._output_path, "w") as f:
            json.dump(caption_id_to_preds, f)

    def output(self):
        return luigi.LocalTarget(self._output_path)


class AllSimToPreds(luigi.WrapperTask):
    def requires(self):
        for dataset, model in MODEL_PREDICTIONS:
            if dataset == MSRVTT:
                yield MsvttSimToPreds(model=model)
            elif dataset == MSVD:
                yield MsvdSimToPreds(model=model)
            else:
                raise ValueError("Invalid dataset")


class MsvdSimToPreds(luigi.Task):
    model = luigi.ChoiceParameter(choices=MODELS)

    def requires(self):
        yield MsvdData()
        yield SimilarityMatrix(model=self.model, dataset="msvd")

    @property
    def _output_path(self):
        base_path = Path("data/predictions")
        return base_path / "msvd" / self.model / "predictions_test.json"

    def run(self):
        sim_matrix_path = Path("data/predictions/msvd") / self.model / "sim_test.npy"
        if self.model == COLLAB_EXPERTS:
            meta_path = Path("data/predictions/msvd") / self.model / "meta.pkl"
        else:
            meta_path = None
        caption_id_to_preds = MsvdEvaluation(
            sim_matrix_path=sim_matrix_path, data_dir="data"
        ).sim_matrix_to_readable(ce_meta_path=meta_path)
        with open(self._output_path, "w") as f:
            json.dump(caption_id_to_preds, f)

    def output(self):
        return luigi.LocalTarget(self._output_path)


HALO_DS = "2022-05-23"
HALO_DATASET_EXPORT = f"data/internal/halo_export_production_{HALO_DS}.pkl"
HALO_ANNOTATOR_MAP = f"data/internal/halo_annotator_id_map_{HALO_DS}.json"


class ExportedHaloData(luigi.ExternalTask):
    """
    This declares internal data dependencies. This data will nto be released, but the code
    processing it into the public versions of the data will be. The primary reason
    this cannot be directly released is that it contains internal annotator ids which
    we cannot release, instead we release a version that remaps those IDs.
    """

    def output(self):
        return [
            luigi.LocalTarget(HALO_DATASET_EXPORT),
            luigi.LocalTarget(HALO_ANNOTATOR_MAP),
        ]


def clean_video_id(video_id: str):
    """
    Videos from halo, even those that were avi, were converted to mp4 for annotation.
    """
    return video_id.replace(".mp4", "")


QUEUE_INFO = {
    "MSVD:CE": (11778, 5090869977623311),
    "MSVD:CLIP4CLIP": (11559, 4785654314865712),
    "MSR-VTT:SSB": (11550, 4995314980504271),
    "MSR-VTT:CE": (11547, 4985131491553502),
    "MSR-VTT:CLIP4CLIP": (11451, 5083898558315210),
    "MSVD:CE-FIXED": (12087, 5142858459131560),
}
QUEUE_TO_MODEL = {
    5090869977623311: ModelEnum.CE,
    4785654314865712: ModelEnum.CLIP4CLIP,
    4995314980504271: ModelEnum.SSB,
    4985131491553502: ModelEnum.CE,
    5083898558315210: ModelEnum.CLIP4CLIP,
    5142858459131560: ModelEnum.CE,
}
QUEUE_TO_DATASET = {
    5090869977623311: DatasetEnum.MSVD,
    4785654314865712: DatasetEnum.MSVD,
    5142858459131560: DatasetEnum.MSVD,
    4995314980504271: DatasetEnum.MSRVTT,
    4985131491553502: DatasetEnum.MSRVTT,
    5083898558315210: DatasetEnum.MSRVTT,
}


class ProcessHaloAnnotations(luigi.Task):
    def requires(self):
        yield ExportedHaloData()

    def run(self):
        annotator_map = {int(k): v for k, v in read_json(HALO_ANNOTATOR_MAP).items()}
        df = pd.read_pickle(HALO_DATASET_EXPORT)
        dataset_annotations: List[FireAnnotation] = []
        disagreements = []
        filtered_df = df[df.video_id != None]
        for (query, video_id), group_df in filtered_df.groupby(["query", "video_id"]):
            video_id = clean_video_id(video_id)
            retrieval_datasets = list(
                {QUEUE_TO_DATASET[qid] for qid in group_df["queue_id"].values}
            )
            if len(retrieval_datasets) != 1:
                raise ValueError()
            else:
                retrieval_dataset = retrieval_datasets[0]
            if len(group_df) == 1:
                row = group_df.iloc[0]
                dataset_annotations.append(
                    FireAnnotation(
                        annotator_ids=[annotator_map[row.annotator_id.item()]],
                        label=row.label,
                        video_id=video_id,
                        query=query,
                        annotator_labels=[row.label],
                        response_ids=[row.response_id],
                        queue_ids=[row.queue_id],
                        models=[QUEUE_TO_MODEL[row.queue_id]],
                        dataset=retrieval_dataset,
                    )
                )
            else:
                modes, counts = stats.mode(group_df["label"].values)

                annotator_ids = [
                    annotator_map[r.item()] for r in group_df["annotator_id"].values
                ]
                pool_models = [
                    QUEUE_TO_MODEL[qid] for qid in group_df["queue_id"].values
                ]
                annotator_labels = group_df["label"].values.tolist()
                response_ids = list(group_df["response_id"].values)
                queue_ids = [int(qid) for qid in group_df["queue_id"].values]
                if counts[0] > (len(group_df) / 2):
                    dataset_annotations.append(
                        FireAnnotation(
                            annotator_ids=annotator_ids,
                            label=modes[0],
                            video_id=video_id,
                            query=query,
                            annotator_labels=annotator_labels,
                            response_ids=response_ids,
                            queue_ids=queue_ids,
                            models=pool_models,
                            dataset=retrieval_dataset,
                        )
                    )
                else:
                    disagreements.append(
                        FireDisagreement(
                            annotator_ids=annotator_ids,
                            video_id=video_id,
                            query=query,
                            annotator_labels=annotator_labels,
                            response_ids=response_ids,
                            queue_ids=queue_ids,
                            models=pool_models,
                            dataset=retrieval_dataset,
                        )
                    )
        fire_dataset = FireDataset(
            annotations=dataset_annotations,
            disagreements=disagreements,
            source_ds=HALO_DS,
            created_ds=str(datetime.datetime.now()),
            source_data_md5=md5sum(HALO_DATASET_EXPORT),
            source_annotator_map_md5=md5sum(HALO_ANNOTATOR_MAP),
            email="me@pedro.ai",
            creator="Pedro Rodriguez",
            website="https://www.pedro.ai/",
            license="Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0), https://creativecommons.org/licenses/by-nc/2.0/",
            datasets=["MSVD", "MSRVTT"],
        )
        fire_msvd_dataset = FireDataset(
            annotations=[
                a for a in dataset_annotations if a.dataset == DatasetEnum.MSVD
            ],
            disagreements=[d for d in disagreements if d.dataset == DatasetEnum.MSVD],
            source_ds=HALO_DS,
            created_ds=str(datetime.datetime.now()),
            source_data_md5=md5sum(HALO_DATASET_EXPORT),
            source_annotator_map_md5=md5sum(HALO_ANNOTATOR_MAP),
            email="me@pedro.ai",
            creator="Pedro Rodriguez",
            website="https://www.pedro.ai/",
            license="Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0), https://creativecommons.org/licenses/by-nc/2.0/",
            datasets=["MSVD"],
        )
        fire_msrvtt_dataset = FireDataset(
            annotations=[
                a for a in dataset_annotations if a.dataset == DatasetEnum.MSRVTT
            ],
            disagreements=[d for d in disagreements if d.dataset == DatasetEnum.MSRVTT],
            source_ds=HALO_DS,
            created_ds=str(datetime.datetime.now()),
            source_data_md5=md5sum(HALO_DATASET_EXPORT),
            source_annotator_map_md5=md5sum(HALO_ANNOTATOR_MAP),
            email="me@pedro.ai",
            creator="Pedro Rodriguez",
            website="https://www.pedro.ai/",
            license="Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0), https://creativecommons.org/licenses/by-nc/2.0/",
            datasets=["MSRVTT"],
        )
        write_json("data/fire_aggregated_dataset.json", fire_dataset)
        write_json("data/fire_msvd_dataset.json", fire_msvd_dataset)
        write_json("data/fire_msrvtt_dataset.json", fire_msrvtt_dataset)

    def output(self):
        return [
            luigi.LocalTarget("data/fire_aggregated_dataset.json"),
            luigi.LocalTarget("data/fire_msvd_dataset.json"),
            luigi.LocalTarget("data/fire_msrvtt_dataset.json"),
        ]


class MsrvttModelEvaluation(luigi.Task):
    model = luigi.Parameter()

    def requires(self):
        yield MsrvttData()
        yield SimilarityMatrix(model=self.model, dataset=MSRVTT)
        yield ProcessHaloAnnotations()

    def run(self):
        sim_path = Path("data/predictions/msrvtt") / self.model / "sim_test.npy"
        evaluation = MsrvttEvaluation(sim_matrix_path=sim_path, data_dir="data")
        captions_to_predictions = evaluation.score_model(
            crowd_annotations_path="data/fire_msrvtt_dataset.json"
        )

        original_rows = []
        augmented_rows = []
        for r in captions_to_predictions.values():
            original_rows.append(r["metrics"])
            augmented_rows.append(r["metrics_annotations"])

        console.log("Original Metrics")
        orig_df = pd.DataFrame(original_rows)
        console.log(orig_df.mean())
        console.log("Fire Metrics")
        fire_df = pd.DataFrame(augmented_rows)
        console.log(fire_df.mean())

        base_path = Path("data/evaluations/msrvtt") / self.model
        base_path.mkdir(exist_ok=True, parents=True)
        orig_df.to_feather(base_path / "original.feather")
        fire_df.to_feather(base_path / "fire.feather")
        write_json(base_path / "captions_to_predictions.json", captions_to_predictions)

    def output(self):
        base_path = Path("data/evaluations/msrvtt") / self.model
        return [
            luigi.LocalTarget(base_path / "original.feather"),
            luigi.LocalTarget(base_path / "fire.feather"),
            luigi.LocalTarget(base_path / "captions_to_predictions.json"),
        ]


class MsvdModelEvaluation(luigi.Task):
    model = luigi.Parameter()

    def requires(self):
        yield MsvdData()
        yield SimilarityMatrix(model=self.model, dataset=MSVD)
        yield ProcessHaloAnnotations()

    def run(self):
        sim_path = Path("data/predictions/msvd") / self.model / "sim_test.npy"
        evaluation = MsvdEvaluation(sim_matrix_path=sim_path, data_dir="data")
        if self.model == COLLAB_EXPERTS:
            captions_to_predictions = evaluation.score_model(
                crowd_annotations_path="data/fire_msvd_dataset.json",
                ce_meta_path="data/predictions/msvd/collaborative-experts/meta.pkl",
            )
        else:
            captions_to_predictions = evaluation.score_model(
                crowd_annotations_path="data/fire_msvd_dataset.json"
            )

        original_rows = []
        augmented_rows = []
        for _, cap_to_result in captions_to_predictions.items():
            for r in cap_to_result.values():
                original_rows.append(r["metrics"])
                augmented_rows.append(r["metrics_annotations"])

        console.log("Original Metrics")
        orig_df = pd.DataFrame(original_rows)
        console.log(orig_df.mean())
        console.log("Fire Metrics")
        fire_df = pd.DataFrame(augmented_rows)
        console.log(fire_df.mean())

        base_path = Path("data/evaluations/msvd") / self.model
        base_path.mkdir(exist_ok=True, parents=True)
        orig_df.to_feather(base_path / "original.feather")
        fire_df.to_feather(base_path / "fire.feather")
        write_json(base_path / "captions_to_predictions.json", captions_to_predictions)

    def output(self):
        base_path = Path("data/evaluations/msvd") / self.model
        return [
            luigi.LocalTarget(base_path / "original.feather"),
            luigi.LocalTarget(base_path / "fire.feather"),
            luigi.LocalTarget(base_path / "captions_to_predictions.json"),
        ]


class AllModelEvaluations(luigi.WrapperTask):
    def requires(self):
        for dataset, model in MODEL_PREDICTIONS:
            if dataset == MSRVTT:
                yield MsrvttModelEvaluation(model=model)
            elif dataset == MSVD:
                yield MsvdModelEvaluation(model=model)
            else:
                raise ValueError()


AUTOMATIC_COMBOS = [
    (CHAR, PERCENTILE, 99, True),
    (CHAR, PERCENTILE, 95, True),
    (CHAR, PERCENTILE, 90, True),
    # (WORD, PERCENTILE, 99, True),
    # (EDIT, PERCENTILE, 99, False),
    # (WORD, PERCENTILE, 95, True),
    # (EDIT, PERCENTILE, 95, False),
    # (WORD, PERCENTILE, 90, True),
    # (EDIT, PERCENTILE, 90, False),
    (CHAR, HIGH_SCORE, 0.75, True),
    (EDIT, HIGH_SCORE, 4, False),
    # (WORD, HIGH_SCORE, 0.95, True),
]


class AutomaticMsrvttMatchingLabels(luigi.Task):
    matcher_type = luigi.Parameter()
    higher_is_better = luigi.BoolParameter(significant=False)
    threshold_type = luigi.Parameter()
    threshold = luigi.FloatParameter()

    def requires(self):
        return MsrvttData()

    @property
    def annotations_path(self):
        return f"data/automatic/annotations_{self.matcher_type}_{self.threshold_type}_{self.threshold}.pickle"

    @property
    def candidates_path(self):
        return f"data/automatic/candidates_{self.matcher_type}_{self.threshold_type}_{self.threshold}.pickle"

    def run(self):
        annotations, candidates = create_msrvtt_automatic_annotations(
            matcher_type=self.matcher_type,
            threshold_type=self.threshold_type,
            threshold=self.threshold,
            higher_is_better=self.higher_is_better,
        )
        with open(
            safe_file(self.annotations_path),
            "wb",
        ) as f:
            pickle.dump(annotations, f)

        with open(
            safe_file(self.candidates_path),
            "wb",
        ) as f:
            pickle.dump(candidates, f)

    def output(self):
        return [
            luigi.LocalTarget(self.annotations_path),
            luigi.LocalTarget(self.candidates_path),
        ]


class AllAutomaticMsrvttMatchingLabels(luigi.WrapperTask):
    def requires(self):
        for (
            matcher_type,
            threshold_type,
            threshold,
            higher_is_better,
        ) in AUTOMATIC_COMBOS:
            yield AutomaticMsrvttMatchingLabels(
                matcher_type=matcher_type,
                higher_is_better=higher_is_better,
                threshold_type=threshold_type,
                threshold=threshold,
            )


class AutomaticMsrvttEval(luigi.Task):
    matcher_type = luigi.Parameter()
    higher_is_better = luigi.BoolParameter()
    threshold_type = luigi.Parameter()
    threshold = luigi.FloatParameter()
    model = luigi.Parameter()

    @property
    def annotation_path(self):
        return f"data/automatic/annotations_{self.matcher_type}_{self.threshold_type}_{self.threshold}.pickle"

    def requires(self):
        yield AutomaticMsrvttMatchingLabels(
            matcher_type=self.matcher_type,
            threshold_type=self.threshold_type,
            threshold=self.threshold,
            higher_is_better=self.higher_is_better,
        )

    @property
    def out_base_path(self):
        return (
            Path("data/automatic/evaluations/msrvtt")
            / self.model
            / f"param_{self.matcher_type}_{self.threshold_type}_{self.threshold}"
        )

    def run(self):
        sim_path = Path("data/predictions/msrvtt") / self.model / "sim_test.npy"
        evaluation = MsrvttEvaluation(sim_matrix_path=sim_path, data_dir="data")
        with open(self.annotation_path, "rb") as f:
            crowd_annotations = pickle.load(f)
        captions_to_predictions = evaluation.score_model(
            crowd_annotations=crowd_annotations
        )

        original_rows = []
        augmented_rows = []
        for r in captions_to_predictions.values():
            original_rows.append(r["metrics"])
            augmented_rows.append(r["metrics_annotations"])

        console.log("Original Metrics")
        orig_df = pd.DataFrame(original_rows)
        console.log(orig_df.mean())
        console.log("Automatic Metrics")
        auto_df = pd.DataFrame(augmented_rows)
        console.log(auto_df.mean())

        self.out_base_path.mkdir(exist_ok=True, parents=True)
        orig_df.to_feather(self.out_base_path / "original.feather")
        auto_df.to_feather(self.out_base_path / "auto.feather")
        write_json(
            self.out_base_path / "captions_to_predictions.json", captions_to_predictions
        )

    def output(self):
        yield luigi.LocalTarget(self.out_base_path / "original.feather")
        yield luigi.LocalTarget(self.out_base_path / "auto.feather")
        yield luigi.LocalTarget(self.out_base_path / "captions_to_predictions.json")


class AllAutomaticMsrvttEvals(luigi.WrapperTask):
    def requires(self):
        for model in [CLIP4CLIP, COLLAB_EXPERTS, SSB]:
            for (
                matcher_type,
                threshold_type,
                threshold,
                higher_is_better,
            ) in AUTOMATIC_COMBOS:
                yield AutomaticMsrvttEval(
                    matcher_type=matcher_type,
                    higher_is_better=higher_is_better,
                    threshold_type=threshold_type,
                    threshold=threshold,
                    model=model,
                )


class ComputeMsrvttTextOverlap(luigi.Task):
    def requires(self):
        yield AllModelEvaluations()

    def run(self):
        train_test_overlap_analysis("data", "data/overlap", k=10)
        train_test_overlap_analysis("data", "data/overlap", k=5)

    def output(self):
        yield luigi.LocalTarget("data/text_similarity_10.feather")
        yield luigi.LocalTarget("data/text_similarity_5.feather")
