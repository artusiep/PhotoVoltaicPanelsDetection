import argparse
import os
from pathlib import Path


def dir_path(path):
    """
    'Type' for argparse - checks that dir exists.
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")


def dir_path_to_create(path):
    """
    'Type' for argparse - checks that dir exists.
    """
    pure_path = Path(path)
    if pure_path.is_dir():
        return path
    else:
        pure_path.mkdir(parents=True, exist_ok=True)
        return path


def extant_file(path):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path
