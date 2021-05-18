import glob
import subprocess

if __name__ == '__main__':
    file_paths = glob.glob("../data/raw/*.JPG")
    for file_path in file_paths:
        file_path
        args = ["exiftool", "-ee", "-b", file_path, "-o", "../data/raw-visible/"]
        subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
