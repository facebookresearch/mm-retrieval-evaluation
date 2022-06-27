import subprocess
import multiprocessing
import json
from pathlib import Path
import typer


def list_videos(data_dir: str):
    with open(Path(data_dir) / "msvd_data/test_list.txt") as f:
        video_ids = [l.strip() for l in f]

    return video_ids


def convert_video(args):
    video_id, video_dir = args
    base_avi_path = Path(video_dir) / "YouTubeClips/"
    base_mp4_path = Path(video_dir) / "YouTubeClipsMP4/"
    avi_path = base_avi_path / f"{video_id}.avi"
    mp4_path = base_mp4_path / f"{video_id}.mp4"
    if mp4_path.exists():
        return {"success": True, "video_id": video_id, "message": "Already encoded"}
    else:
        try:
            subprocess.run(f"ffmpeg -i {avi_path} {mp4_path}", shell=True, check=True)
            return {
                "success": True,
                "video_id": video_id,
                "message": "Success encoding",
            }
        except:
            return {"success": False, "video_id": video_id, "message": ""}


app = typer.Typer()


@app.command()
def main(data_dir: str, video_dir: str):
    with multiprocessing.Pool() as p:
        videos = [(v, video_dir) for v in list_videos(data_dir)]
        statuses = p.map(convert_video, videos)

    with open("conversion_status.json", "w") as f:
        json.dump(statuses, f)


if __name__ == "__main__":
    app()
