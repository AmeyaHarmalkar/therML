import subprocess
from pathlib import Path

LOG_DIR = Path(__file__).absolute().parent.with_name("logs")
SINGER_LOG_DIR = LOG_DIR.with_name("singer_logs")


def sync_upload(include_ckpt: bool = False):
    command = [
        "aws",
        "s3",
        "sync",
        str(LOG_DIR),
        "s3://stability-prediction/supervised/",
    ]
    if not include_ckpt:
        command.extend([
            "--exclude",
            "*.ckpt",
        ])
    subprocess.run(command)


def sync_download(include_ckpt: bool = False):
    command = [
        "aws",
        "s3",
        "sync",
        "s3://stability-prediction/supervised/",
        str(LOG_DIR),
    ]
    if not include_ckpt:
        command.extend([
            "--exclude",
            "*.ckpt",
        ])
    subprocess.run(command)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_upload = subparsers.add_parser("upload")
    parser_upload.add_argument("--include_ckpt", action="store_true")
    parser_upload.set_defaults(func=sync_upload)
    parser_download = subparsers.add_parser("download")
    parser_download.add_argument("--include_ckpt", action="store_true")
    parser_download.set_defaults(func=sync_download)

    args = parser.parse_args()
    args.func(args.include_ckpt)
