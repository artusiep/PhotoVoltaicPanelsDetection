import argparse
import os


def dir_path(path):
    """
    'Type' for argparse - checks that dir exists.
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")


def extant_file(path):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path
