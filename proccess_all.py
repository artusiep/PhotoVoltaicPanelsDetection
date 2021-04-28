import glob
from datetime import datetime

from main import main, save_img

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    file_paths = glob.glob("data/thermal/*.JPG")
    for index, file_path in enumerate(file_paths):
        new_file_path = file_path.replace('thermal', f'annotated/{current_time}')
        print(f"Annotation of img {file_path} started. Result will be saved to {new_file_path}. Run index {index}")
        try:
            annotated_img = main(file_path)
        except Exception as e:
            print(f"Annotation of img {file_path} failed with {e}")
            continue
        save_img(annotated_img, new_file_path)
