import flirimageextractor
import matplotlib
from exif import Image

file_path = '../data/raw/DJI_4_R(600).JPG'

with open(file_path, 'rb') as image_file:
    my_image = Image(image_file)
    # print(my_image.has_exif)

palettes = [matplotlib.jet, matplotlib.bwr, matplotlib.gist_ncar, matplotlib.plasma, matplotlib.turbo]
flir = flirimageextractor.FlirImageExtractor(palettes=palettes)
flir.process_image(file_path)

flir.save_images()
print("Finished")
