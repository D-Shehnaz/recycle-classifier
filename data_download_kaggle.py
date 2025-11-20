# data_download_kaggle.py
# Usage:
#   python data_download_kaggle.py --dataset <kaggle/dataset-slug> --target dataset
# Example:
#   python data_download_kaggle.py --dataset "gtox/trashnet" --target dataset

import argparse
import os
import zipfile
import subprocess
import sys

def run_kaggle_download(dataset, target):
    # Requires kaggle CLI installed and configured (pip install kaggle)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", target, "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="kaggle dataset slug e.g. username/dataset-name")
    parser.add_argument("--target", default="dataset", help="where to put downloaded dataset")
    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)
    try:
        run_kaggle_download(args.dataset, args.target)
        print("Downloaded and extracted into", args.target)
    except Exception as e:
        print("Error downloading dataset via Kaggle CLI:", e)
        print("Make sure the Kaggle CLI is installed and %USERPROFILE%/.kaggle/kaggle.json exists.")

if __name__ == "__main__":
    main()
