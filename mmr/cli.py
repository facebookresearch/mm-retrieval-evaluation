import typer
import json
import pandas as pd
from mmr.analysis import MsrvttEvaluation, MsvdEvaluation


app = typer.Typer()
msrvtt_cli = typer.Typer()
msvd_cli = typer.Typer()
app.add_typer(msrvtt_cli, name="msrvtt")
app.add_typer(msvd_cli, name="msvd")


@msrvtt_cli.command()
def evaluate(sim_matrix_path: str, output_path: str):
    evaluation = MsrvttEvaluation(sim_matrix_path=sim_matrix_path, data_dir="data")
    captions_to_predictions = evaluation.score_model(
        crowd_annotations_path="data/fire_msrvtt_dataset.json",
        # analysis_annotations_path="data/analysis_par_annotations.json"
    )
    with open(output_path, "w") as f:
        json.dump(captions_to_predictions, f)

    rows = []
    for row in captions_to_predictions.values():
        rows.append(row["metrics"])
    print("Original annotations")
    print(pd.DataFrame(rows).mean())

    print("Augmented Annotations")
    rows = []
    for row in captions_to_predictions.values():
        rows.append(row["metrics_annotations"])
    print(pd.DataFrame(rows).mean())


@msrvtt_cli.command()
def readable_preds(sim_matrix_path: str, output_path: str):
    caption_id_to_preds = MsrvttEvaluation(
        sim_matrix_path=sim_matrix_path, data_dir="data"
    ).sim_matrix_to_readable()
    with open(output_path, "w") as f:
        json.dump(caption_id_to_preds, f)


@msvd_cli.command()
def readable_preds(sim_matrix_path: str, output_path: str):
    caption_id_to_preds = MsvdEvaluation(
        sim_matrix_path=sim_matrix_path, data_dir="data"
    ).sim_matrix_to_readable()
    with open(output_path, "w") as f:
        json.dump(caption_id_to_preds, f)


if __name__ == "__main__":
    app()
